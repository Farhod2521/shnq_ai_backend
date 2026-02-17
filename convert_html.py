import argparse
import os
import re

from bs4 import BeautifulSoup

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402

django.setup()  # noqa: E402

from django.db import transaction  # noqa: E402
from app_shnq.models import (  # noqa: E402
    Category,
    Chapter,
    Clause,
    Document,
    NormImage,
    NormTable,
    NormTableCell,
    NormTableRow,
    ensure_runtime_tables,
)
from app_shnq.table_i18n import ensure_normtable_i18n_columns, pretranslate_table_fields  # noqa: E402


MIN_TEXT_LEN = 30
TABLE_NUMBER_PATTERNS = [
    re.compile(r"\bjadval\s*[-.]?\s*([0-9]+[a-z]?)\b", re.IGNORECASE),
    re.compile(r"\b([0-9]+[a-z]?)\s*[-.]?\s*jadval\b", re.IGNORECASE),
]
APPENDIX_NUMBER_PATTERN = re.compile(
    r"(?:\b(\d+)\s*[-.]?\s*ilova(?:si|da|ga|dan|ning|lar)?\b|"
    r"\bilova(?:si|da|ga|dan|ning|lar)?\s*[-.]?\s*(\d+)\b)",
    re.IGNORECASE,
)


def clean_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_clause_number(text):
    match = re.match(r"^(\d{1,3}(?:\.\d{1,3})*)\s*[\).:-]?\s*", text)
    return match.group(1) if match else None


def extract_doc_code_from_html(soup):
    title_node = soup.find("div", class_=re.compile(r"ACT_TITLE|ACT_TITLE_APPL"))
    title_text = clean_text(title_node.get_text(" ", strip=True)) if title_node else ""
    match = re.search(r"\b(SHNQ|QMQ|KMK|SNIP)\s+([0-9][0-9.\-]*)\b", title_text, re.IGNORECASE)
    if not match:
        return None
    return f"{match.group(1).upper()} {match.group(2)}"


def _extract_table_number(text):
    cleaned = clean_text(text).lower()
    for pattern in TABLE_NUMBER_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            return match.group(1)
    return None


def _extract_table_title(text):
    cleaned = clean_text(text)
    lowered = cleaned.lower()
    for pattern in TABLE_NUMBER_PATTERNS:
        match = pattern.search(lowered)
        if match:
            start, end = match.span()
            candidate = (cleaned[:start] + cleaned[end:]).strip(" -:.")
            return candidate or cleaned
    return cleaned


def _extract_appendix_number(text):
    cleaned = clean_text(text).lower()
    match = APPENDIX_NUMBER_PATTERN.search(cleaned)
    if not match:
        return None
    return match.group(1) or match.group(2)


def _safe_int(value, default=1):
    try:
        num = int(value)
        return num if num > 0 else default
    except (TypeError, ValueError):
        return default


def _escape_md(text):
    return (text or "").replace("|", r"\|")


def _extract_table_label(table_elem):
    caption = table_elem.find("caption")
    if caption:
        caption_text = clean_text(caption.get_text(" ", strip=True))
        if caption_text:
            return caption_text

    sibling = table_elem
    for _ in range(4):
        sibling = sibling.find_previous_sibling()
        if not sibling:
            break
        sibling_text = clean_text(sibling.get_text(" ", strip=True))
        sibling_classes = {cls.upper() for cls in (sibling.get("class") or [])}
        if sibling_text and (
            "jadval" in sibling_text.lower()
            or _extract_table_number(sibling_text)
            or _extract_appendix_number(sibling_text)
            or "ACT_TITLE_APPL" in sibling_classes
            or any(cls.startswith("APPL_BANNER") for cls in sibling_classes)
        ):
            return sibling_text
    return ""


def _extract_image_context(img_elem):
    parent_div = img_elem.find_parent("div")
    candidates = []

    def add_text(node):
        if not node or node.find("img"):
            return
        text = clean_text(node.get_text(" ", strip=True))
        if not text:
            return
        if text.lower().startswith(("http://", "https://")):
            return
        candidates.append(text[:220])

    if parent_div:
        add_text(parent_div)

        prev = parent_div
        for _ in range(3):
            prev = prev.find_previous_sibling("div")
            if not prev:
                break
            add_text(prev)
            prev_classes = {cls.upper() for cls in (prev.get("class") or [])}
            if (
                "TEXT_BOLD_CENTER" in prev_classes
                or "ACT_TITLE_APPL" in prev_classes
                or "TEXT_HEADER_DEFAULT" in prev_classes
                or any(cls.startswith("APPL_BANNER") for cls in prev_classes)
            ):
                break

        nxt = parent_div
        for _ in range(2):
            nxt = nxt.find_next_sibling("div")
            if not nxt:
                break
            add_text(nxt)
            nxt_classes = {cls.upper() for cls in (nxt.get("class") or [])}
            if "TEXT_CENTER" in nxt_classes or "TEXT_BOLD_CENTER" in nxt_classes:
                break

    unique = []
    seen = set()
    for text in candidates:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(text)
        if len(unique) >= 3:
            break
    return " | ".join(unique)


def _table_to_rows_and_markdown(table_elem):
    parsed_rows = []
    expanded_rows = []
    pending = {}

    for tr in table_elem.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue

        row_values = []
        row_cells = []
        col_index = 1

        def flush_pending():
            nonlocal col_index
            while col_index in pending:
                text, is_header, remain = pending[col_index]
                row_values.append(text)
                if remain > 1:
                    pending[col_index] = (text, is_header, remain - 1)
                else:
                    pending.pop(col_index, None)
                col_index += 1

        flush_pending()
        for cell in cells:
            flush_pending()
            text = clean_text(cell.get_text(" ", strip=True))
            is_header = cell.name == "th"
            row_span = _safe_int(cell.get("rowspan"), default=1)
            col_span = _safe_int(cell.get("colspan"), default=1)

            row_cells.append(
                {
                    "col_index": col_index,
                    "text": text,
                    "is_header": is_header,
                    "row_span": row_span,
                    "col_span": col_span,
                }
            )

            for offset in range(col_span):
                row_values.append(text)
                if row_span > 1:
                    pending[col_index + offset] = (text, is_header, row_span - 1)
            col_index += col_span

        flush_pending()
        parsed_rows.append(row_cells)
        expanded_rows.append(row_values)

    if not expanded_rows:
        return parsed_rows, ""

    col_count = max(len(row) for row in expanded_rows)
    normalized = [row + [""] * (col_count - len(row)) for row in expanded_rows]
    header = normalized[0]
    sep = ["---"] * col_count
    markdown_lines = [
        "| " + " | ".join(_escape_md(col) for col in header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in normalized[1:]:
        markdown_lines.append("| " + " | ".join(_escape_md(col) for col in row) + " |")

    return parsed_rows, "\n".join(markdown_lines)


@transaction.atomic
def import_shnq_html(
    file_path,
    category_code="SHNQ",
    doc_code="SHNQ",
    title="SHNQ",
    lex_url=None,
    reset=True,
    images_only=False,
):
    ensure_normtable_i18n_columns()

    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    inferred_code = extract_doc_code_from_html(soup)
    effective_doc_code = inferred_code or doc_code
    inferred_category_code = (effective_doc_code.split(" ", 1)[0].upper() if effective_doc_code else category_code)
    effective_category_code = (
        inferred_category_code
        if re.search(r"\d", (category_code or ""))
        else (category_code or inferred_category_code)
    )

    category, _ = Category.objects.get_or_create(
        code=effective_category_code, defaults={"name": effective_category_code}
    )
    existing_doc = (
        Document.objects.filter(code=effective_doc_code)
        .select_related("category")
        .order_by("created_at")
        .first()
    )
    if existing_doc:
        document = existing_doc
        created = False
        if document.category_id != category.id:
            document.category = category
            document.save(update_fields=["category"])
    else:
        document, created = Document.objects.get_or_create(
            category=category,
            code=effective_doc_code,
            defaults={"title": title, "lex_url": lex_url},
        )
    if not created:
        changed = False
        if title and document.title != title:
            document.title = title
            changed = True
        if lex_url != document.lex_url:
            document.lex_url = lex_url
            changed = True
        if changed:
            document.save(update_fields=["title", "lex_url"])

    if reset:
        NormImage.objects.filter(document=document).delete()
        if not images_only:
            NormTable.objects.filter(document=document).delete()
            Clause.objects.filter(document=document).delete()
            Chapter.objects.filter(document=document).delete()

    current_chapter = None
    chapter_order = 0
    clause_order = 0
    table_order = 0
    table_seq = 0
    image_order = 0
    pending_anchor = None
    last_clause = None
    last_table = None
    last_image = None
    current_appendix_number = None
    current_appendix_title = None
    imported_tables = 0
    imported_clauses = 0
    imported_images = 0
    seen_image_sources = set()

    for elem in soup.find_all(["div", "a", "table", "img"]):
        if elem.name == "a" and elem.get("id"):
            anchor = elem.get("id")
            if elem.find("img"):
                pending_anchor = anchor
                continue
            if last_table and not last_table.html_anchor:
                last_table.html_anchor = anchor
                last_table.save(update_fields=["html_anchor"])
            elif last_clause and not last_clause.html_anchor:
                last_clause.html_anchor = anchor
                last_clause.save(update_fields=["html_anchor"])
            elif last_image and not last_image.html_anchor:
                last_image.html_anchor = anchor
                last_image.save(update_fields=["html_anchor"])
            else:
                pending_anchor = anchor
            continue

        if elem.name == "div" and "TEXT_HEADER_DEFAULT" in elem.get("class", []):
            header_text = clean_text(elem.get_text())
            if header_text:
                chapter_order += 1
                current_chapter = Chapter.objects.create(
                    document=document,
                    title=header_text,
                    order=chapter_order,
                )
                current_appendix_number = None
                current_appendix_title = None
            continue

        elem_classes = {cls.upper() for cls in (elem.get("class") or [])}
        if elem.name == "div" and any(cls.startswith("APPL_BANNER") for cls in elem_classes):
            banner_text = clean_text(elem.get_text(" ", strip=True))
            appendix_number = _extract_appendix_number(banner_text)
            if appendix_number:
                current_appendix_number = appendix_number
            continue

        if elem.name == "div" and "ACT_TITLE_APPL" in elem_classes:
            current_appendix_title = clean_text(elem.get_text(" ", strip=True)) or None
            continue

        if elem.name == "div" and "ACT_TEXT" in elem.get("class", []):
            if images_only:
                continue
            text = clean_text(elem.get_text())
            if len(text) < MIN_TEXT_LEN:
                continue

            clause_number = extract_clause_number(text)
            clause_order += 1
            last_clause = Clause.objects.create(
                document=document,
                chapter=current_chapter,
                clause_number=clause_number,
                html_anchor=pending_anchor,
                text=text,
                order=clause_order,
            )
            pending_anchor = None
            imported_clauses += 1
            continue

        if elem.name == "img":
            src = clean_text(elem.get("src") or "")
            if not src or src in seen_image_sources:
                pending_anchor = None
                continue

            image_anchor = pending_anchor
            if not image_anchor:
                parent_anchor = elem.find_parent("a")
                if parent_anchor and parent_anchor.get("id"):
                    image_anchor = parent_anchor.get("id")

            image_context = _extract_image_context(elem)
            section_title = current_chapter.title if current_chapter else None
            if current_appendix_number:
                appendix_label = f"{current_appendix_number}-ilova"
                section_title = f"{appendix_label}. {current_appendix_title}" if current_appendix_title else appendix_label

            image_order += 1
            last_image = NormImage.objects.create(
                document=document,
                chapter=current_chapter,
                section_title=section_title,
                appendix_number=current_appendix_number,
                title=current_appendix_title,
                html_anchor=image_anchor,
                image_url=src,
                context_text=image_context or "",
                order=image_order,
            )
            pending_anchor = None
            imported_images += 1
            seen_image_sources.add(src)
            continue

        if elem.name == "table":
            if images_only:
                continue
            table_label = _extract_table_label(elem)
            table_number = _extract_table_number(table_label or "")
            if not table_number:
                table_number = _extract_table_number(clean_text(elem.get_text(" ", strip=True)[:120]))
            appendix_number = _extract_appendix_number(table_label or "") or current_appendix_number
            if not table_number and appendix_number:
                table_number = f"ilova-{appendix_number}"
            if not table_number:
                table_seq += 1
                table_number = str(table_seq)
            table_title = _extract_table_title(table_label) if table_label else None
            if (not table_title or table_title.lower() == f"{appendix_number}-ilova") and current_appendix_title:
                table_title = current_appendix_title
            raw_html = str(elem)
            rows, markdown = _table_to_rows_and_markdown(elem)
            if not rows:
                continue

            section_title = current_chapter.title if current_chapter else None
            if appendix_number:
                appendix_label = f"{appendix_number}-ilova"
                section_title = f"{appendix_label}. {current_appendix_title}" if current_appendix_title else appendix_label

            table_order += 1
            last_table = NormTable.objects.create(
                document=document,
                chapter=current_chapter,
                section_title=section_title,
                table_number=table_number,
                title=table_title,
                html_anchor=pending_anchor,
                raw_html=raw_html,
                markdown=markdown,
                order=table_order,
            )
            pretranslate_table_fields(last_table)
            pending_anchor = None
            imported_tables += 1

            for row_idx, row_cells in enumerate(rows, start=1):
                row_obj = NormTableRow.objects.create(table=last_table, row_index=row_idx)
                for cell in row_cells:
                    NormTableCell.objects.create(
                        row=row_obj,
                        col_index=cell["col_index"],
                        text=cell["text"],
                        is_header=cell["is_header"],
                        row_span=cell["row_span"],
                        col_span=cell["col_span"],
                    )

    return {"clauses": imported_clauses, "tables": imported_tables, "images": imported_images}


def _parse_args():
    parser = argparse.ArgumentParser(description="SHNQ HTML faylini DB ga import qilish.")
    parser.add_argument("--file", default="shnq.html", help="HTML fayl yo'li.")
    parser.add_argument("--category-code", default="SHNQ", help="Category code (masalan SHNQ).")
    parser.add_argument("--doc-code", default="SHNQ", help="Hujjat kodi (masalan SHNQ 2.07.01-23).")
    parser.add_argument("--title", default="SHNQ", help="Hujjat nomi.")
    parser.add_argument("--lex-url", default=None, help="Rasmiy hujjat havolasi.")
    parser.add_argument("--no-reset", action="store_true", help="Mavjud document ma'lumotlarini tozalamaslik.")
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Faqat rasmlarni import qilish (Clause/NormTable o'zgarmaydi).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = args.file if os.path.isabs(args.file) else os.path.join(base_dir, args.file)
    ensure_runtime_tables()
    result = import_shnq_html(
        file_path=html_path,
        category_code=args.category_code,
        doc_code=args.doc_code,
        title=args.title,
        lex_url=args.lex_url,
        reset=not args.no_reset,
        images_only=args.images_only,
    )
    print(
        "Import finished. "
        f"Clauses: {result['clauses']}, "
        f"Tables: {result['tables']}, "
        f"Images: {result.get('images', 0)}, "
        f"File: {html_path}"
    )
