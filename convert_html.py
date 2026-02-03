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
    NormTable,
    NormTableCell,
    NormTableRow,
)


MIN_TEXT_LEN = 30
TABLE_NUMBER_PATTERNS = [
    re.compile(r"\bjadval\s*[-.]?\s*([0-9]+[a-z]?)\b", re.IGNORECASE),
    re.compile(r"\b([0-9]+[a-z]?)\s*[-.]?\s*jadval\b", re.IGNORECASE),
]


def clean_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_clause_number(text):
    match = re.match(r"^(\d+(\.\d+)+)", text)
    return match.group(1) if match else None


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
        if sibling_text and ("jadval" in sibling_text.lower() or _extract_table_number(sibling_text)):
            return sibling_text
    return ""


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
):
    category, _ = Category.objects.get_or_create(
        code=category_code, defaults={"name": category_code}
    )
    document, created = Document.objects.get_or_create(
        category=category,
        code=doc_code,
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
        NormTable.objects.filter(document=document).delete()
        Clause.objects.filter(document=document).delete()
        Chapter.objects.filter(document=document).delete()

    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    current_chapter = None
    chapter_order = 0
    clause_order = 0
    table_order = 0
    table_seq = 0
    pending_anchor = None
    last_clause = None
    last_table = None
    imported_tables = 0
    imported_clauses = 0

    for elem in soup.find_all(["div", "a", "table"]):
        if elem.name == "a" and elem.get("id"):
            anchor = elem.get("id")
            if last_table and not last_table.html_anchor:
                last_table.html_anchor = anchor
                last_table.save(update_fields=["html_anchor"])
            elif last_clause and not last_clause.html_anchor:
                last_clause.html_anchor = anchor
                last_clause.save(update_fields=["html_anchor"])
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
            continue

        if elem.name == "div" and "ACT_TEXT" in elem.get("class", []):
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

        if elem.name == "table":
            table_label = _extract_table_label(elem)
            table_number = _extract_table_number(table_label or "")
            if not table_number:
                table_number = _extract_table_number(clean_text(elem.get_text(" ", strip=True)[:120]))
            if not table_number:
                table_seq += 1
                table_number = str(table_seq)
            table_title = _extract_table_title(table_label) if table_label else None
            raw_html = str(elem)
            rows, markdown = _table_to_rows_and_markdown(elem)
            if not rows:
                continue

            table_order += 1
            last_table = NormTable.objects.create(
                document=document,
                chapter=current_chapter,
                section_title=current_chapter.title if current_chapter else None,
                table_number=table_number,
                title=table_title,
                html_anchor=pending_anchor,
                raw_html=raw_html,
                markdown=markdown,
                order=table_order,
            )
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

    return {"clauses": imported_clauses, "tables": imported_tables}


def _parse_args():
    parser = argparse.ArgumentParser(description="SHNQ HTML faylini DB ga import qilish.")
    parser.add_argument("--file", default="shnq.html", help="HTML fayl yo'li.")
    parser.add_argument("--category-code", default="SHNQ", help="Category code (masalan SHNQ).")
    parser.add_argument("--doc-code", default="SHNQ", help="Hujjat kodi (masalan SHNQ 2.07.01-23).")
    parser.add_argument("--title", default="SHNQ", help="Hujjat nomi.")
    parser.add_argument("--lex-url", default=None, help="Rasmiy hujjat havolasi.")
    parser.add_argument("--no-reset", action="store_true", help="Mavjud document ma'lumotlarini tozalamaslik.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = args.file if os.path.isabs(args.file) else os.path.join(base_dir, args.file)
    result = import_shnq_html(
        file_path=html_path,
        category_code=args.category_code,
        doc_code=args.doc_code,
        title=args.title,
        lex_url=args.lex_url,
        reset=not args.no_reset,
    )
    print(
        f"Import finished. Clauses: {result['clauses']}, Tables: {result['tables']}, File: {html_path}"
    )
