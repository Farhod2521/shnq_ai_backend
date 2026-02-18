import os
import re
import time

from django.db import OperationalError

from .deepseek_client import DEFAULT_EMBED_MODEL, embed_text as deepseek_embed_text
from .models import NormTableRow, TableRowEmbedding, ensure_runtime_tables


TABLE_ROW_PROGRESS_EVERY = int(os.getenv("TABLE_ROW_PROGRESS_EVERY", "100"))
SQLITE_LOCK_MAX_RETRIES = int(os.getenv("SQLITE_LOCK_MAX_RETRIES", "12"))
SQLITE_LOCK_RETRY_BASE_SEC = float(os.getenv("SQLITE_LOCK_RETRY_BASE_SEC", "0.25"))


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _retry_on_sqlite_lock(action, fn):
    for attempt in range(SQLITE_LOCK_MAX_RETRIES):
        try:
            return fn()
        except OperationalError as exc:
            msg = str(exc).lower()
            if "database is locked" not in msg:
                raise
            if attempt >= SQLITE_LOCK_MAX_RETRIES - 1:
                raise
            delay = SQLITE_LOCK_RETRY_BASE_SEC * (2 ** attempt)
            print(
                f"[sqlite-lock] {action}: retry {attempt + 1}/{SQLITE_LOCK_MAX_RETRIES} in {delay:.2f}s",
                flush=True,
            )
            time.sleep(delay)


def _extract_table_headers(table):
    rows = list(table.rows.prefetch_related("cells").order_by("row_index"))
    if not rows:
        return {}

    headers = {}
    has_explicit_headers = False
    for row in rows:
        for cell in row.cells.all():
            text = _clean_text(cell.text)
            if not text:
                continue
            if cell.is_header:
                has_explicit_headers = True
                headers[cell.col_index] = text
    if has_explicit_headers:
        return headers

    # Fallback: birinchi mazmunli satrni header deb olamiz.
    for row in rows:
        first_row_values = []
        for cell in row.cells.all():
            text = _clean_text(cell.text)
            if not text:
                continue
            first_row_values.append((cell.col_index, text))
        if not first_row_values:
            continue
        for col_index, text in first_row_values:
            headers[col_index] = text
        break
    return headers


def build_table_row_search_text(row, headers=None):
    table = row.table
    headers = headers or {}

    values = []
    for cell in row.cells.all():
        text = _clean_text(cell.text)
        if not text:
            continue
        header = _clean_text(headers.get(cell.col_index, ""))
        if header and header.lower() != text.lower():
            values.append(f"{header}: {text}")
        else:
            values.append(f"Ustun {cell.col_index}: {text}")

    if not values:
        return ""

    chapter_title = table.section_title or (table.chapter.title if table.chapter else "")
    lines = [
        f"Hujjat: {table.document.code}",
        f"Jadval: {table.table_number}",
        f"Satr: {row.row_index}",
    ]
    if chapter_title:
        lines.append(f"Bo'lim: {chapter_title}")
    if table.title:
        lines.append(f"Sarlavha: {table.title}")
    lines.append("Qiymatlar: " + " | ".join(values))
    return "\n".join(lines)


def upsert_table_row_embeddings(
    embedding_model=None,
    force_update=False,
    limit=None,
    doc_code=None,
    table_number=None,
    table_id=None,
):
    ensure_runtime_tables()
    model_name = embedding_model or DEFAULT_EMBED_MODEL
    qs = NormTableRow.objects.select_related(
        "table",
        "table__document",
        "table__chapter",
    ).prefetch_related("cells").order_by("table__document__code", "table__order", "row_index")

    if table_id:
        qs = qs.filter(table_id=table_id)
    if doc_code:
        qs = qs.filter(table__document__code__iexact=(doc_code or "").strip())
    if table_number:
        qs = qs.filter(table__table_number__iexact=(table_number or "").strip())
    if limit:
        qs = qs[:limit]

    created = 0
    updated = 0
    skipped = 0

    header_cache = {}
    total = qs.count()
    for idx, row in enumerate(qs.iterator(chunk_size=200), start=1):
        cache_key = str(row.table_id)
        headers = header_cache.get(cache_key)
        if headers is None:
            headers = _extract_table_headers(row.table)
            header_cache[cache_key] = headers

        search_text = build_table_row_search_text(row, headers=headers)
        if not search_text.strip():
            skipped += 1
            if TABLE_ROW_PROGRESS_EVERY > 0 and (idx == 1 or idx % TABLE_ROW_PROGRESS_EVERY == 0 or idx == total):
                print(
                    f"[table_rows {idx}/{total}] created={created} updated={updated} skipped={skipped}",
                    flush=True,
                )
            continue

        existing = _retry_on_sqlite_lock(
            f"table_row existing lookup {idx}/{total}",
            lambda: TableRowEmbedding.objects.filter(row=row).first(),
        )
        if (
            existing
            and not force_update
            and existing.embedding_model == model_name
            and existing.search_text == search_text
            and existing.vector
        ):
            skipped += 1
            if TABLE_ROW_PROGRESS_EVERY > 0 and (idx == 1 or idx % TABLE_ROW_PROGRESS_EVERY == 0 or idx == total):
                print(
                    f"[table_rows {idx}/{total}] created={created} updated={updated} skipped={skipped}",
                    flush=True,
                )
            continue

        vector = deepseek_embed_text(search_text, model=model_name)
        token_count = len(search_text.split())
        chapter_title = row.table.section_title or (row.table.chapter.title if row.table.chapter else None)

        _, was_created = _retry_on_sqlite_lock(
            f"table_row upsert {idx}/{total}",
            lambda: TableRowEmbedding.objects.update_or_create(
                row=row,
                defaults={
                    "embedding_model": model_name,
                    "vector": vector,
                    "token_count": token_count,
                    "shnq_code": row.table.document.code,
                    "chapter_title": chapter_title,
                    "table_number": row.table.table_number,
                    "table_title": row.table.title,
                    "row_index": row.row_index,
                    "search_text": search_text,
                },
            ),
        )
        if was_created:
            created += 1
        else:
            updated += 1
        if TABLE_ROW_PROGRESS_EVERY > 0 and (idx == 1 or idx % TABLE_ROW_PROGRESS_EVERY == 0 or idx == total):
            print(
                f"[table_rows {idx}/{total}] created={created} updated={updated} skipped={skipped}",
                flush=True,
            )

    return {"created": created, "updated": updated, "skipped": skipped}


def upsert_table_row_embeddings_for_table(table, embedding_model=None, force_update=False):
    return upsert_table_row_embeddings(
        embedding_model=embedding_model,
        force_update=force_update,
        table_id=str(table.id),
    )
