import os
import time

from django.db import OperationalError
from django.db import connection

from .deepseek_client import translate_html_preserving_tags, translate_text


TABLE_I18N_LANGS = ("ru", "en", "ko")
TABLE_PRETRANSLATE_MODEL = os.getenv("TABLE_PRETRANSLATE_MODEL", "gpt-4o-mini")
SQLITE_LOCK_MAX_RETRIES = int(os.getenv("SQLITE_LOCK_MAX_RETRIES", "12"))
SQLITE_LOCK_RETRY_BASE_SEC = float(os.getenv("SQLITE_LOCK_RETRY_BASE_SEC", "0.25"))

I18N_COLUMNS = {
    "raw_html_ru": "TEXT",
    "raw_html_en": "TEXT",
    "raw_html_ko": "TEXT",
    "markdown_ru": "TEXT",
    "markdown_en": "TEXT",
    "markdown_ko": "TEXT",
}


def ensure_normtable_i18n_columns():
    table_name = "app_shnq_normtable"
    with connection.cursor() as cursor:
        cols = {row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()}
        for col_name, col_type in I18N_COLUMNS.items():
            if col_name in cols:
                continue
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type} NOT NULL DEFAULT ''")


def _save_table_with_retry(table, update_fields):
    for attempt in range(SQLITE_LOCK_MAX_RETRIES):
        try:
            table.save(update_fields=update_fields)
            return
        except OperationalError as exc:
            msg = str(exc).lower()
            if "database is locked" not in msg:
                raise
            if attempt >= SQLITE_LOCK_MAX_RETRIES - 1:
                raise
            delay = SQLITE_LOCK_RETRY_BASE_SEC * (2 ** attempt)
            print(
                f"[sqlite-lock] pretranslate save {table.document.code} {table.table_number}: "
                f"retry {attempt + 1}/{SQLITE_LOCK_MAX_RETRIES} in {delay:.2f}s",
                flush=True,
            )
            time.sleep(delay)


def get_pretranslated_table_content(table, language: str):
    lang = (language or "uz").lower()
    if lang not in TABLE_I18N_LANGS:
        return table.raw_html, table.markdown

    raw_html = getattr(table, f"raw_html_{lang}", "") or table.raw_html
    markdown = getattr(table, f"markdown_{lang}", "") or table.markdown
    return raw_html, markdown


def has_pretranslated_table_content(table, language: str) -> bool:
    lang = (language or "uz").lower()
    if lang not in TABLE_I18N_LANGS:
        return True
    raw_html = getattr(table, f"raw_html_{lang}", "") or ""
    markdown = getattr(table, f"markdown_{lang}", "") or ""
    return bool(raw_html.strip() and markdown.strip())


def pretranslate_table_fields(table, force: bool = False):
    enabled = os.getenv("TABLE_PRETRANSLATE", "1") == "1"
    if not enabled and not force:
        return table

    source_html = table.raw_html or ""
    source_md = table.markdown or ""
    if not source_html and not source_md:
        return table

    changed = []
    for lang in TABLE_I18N_LANGS:
        html_field = f"raw_html_{lang}"
        md_field = f"markdown_{lang}"
        cur_html = getattr(table, html_field, "") or ""
        cur_md = getattr(table, md_field, "") or ""

        if force or not cur_html:
            try:
                value = translate_html_preserving_tags(
                    source_html,
                    target_language=lang,
                    source_language="uz",
                    model=TABLE_PRETRANSLATE_MODEL,
                    timeout=6,
                )
            except Exception:
                value = source_html
            setattr(table, html_field, value or source_html)
            changed.append(html_field)

        if force or not cur_md:
            try:
                value = _translate_large_markdown(source_md, target_language=lang)
            except Exception:
                value = source_md
            setattr(table, md_field, value or source_md)
            changed.append(md_field)

    if changed:
        _save_table_with_retry(table, update_fields=sorted(set(changed)))
    return table


def _translate_large_markdown(text: str, target_language: str) -> str:
    payload = (text or "").strip()
    if not payload:
        return payload
    if len(payload) <= 2200:
        return translate_text(
            payload,
            target_language=target_language,
            source_language="uz",
            model=TABLE_PRETRANSLATE_MODEL,
            timeout=8,
        )

    lines = payload.splitlines()
    chunks = []
    buf = []
    size = 0
    for line in lines:
        part = line if line is not None else ""
        add = len(part) + 1
        if size + add > 1800 and buf:
            chunks.append("\n".join(buf))
            buf = [part]
            size = add
        else:
            buf.append(part)
            size += add
    if buf:
        chunks.append("\n".join(buf))

    out = []
    for chunk in chunks:
        try:
            tr = translate_text(
                chunk,
                target_language=target_language,
                source_language="uz",
                model=TABLE_PRETRANSLATE_MODEL,
                timeout=8,
            )
        except Exception:
            tr = chunk
        out.append(tr or chunk)
    return "\n".join(out)
