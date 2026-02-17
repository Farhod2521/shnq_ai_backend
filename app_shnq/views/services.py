import json
import os
from pathlib import Path

from .constants import *

# NOTE:
# `from .constants import *` does not import underscored names, so keep
# cache state local to this module to avoid NameError at runtime.
_EMBED_CACHE_MODEL = None
_EMBED_CACHE_DATA = None
_IMAGE_EMBED_CACHE_MODEL = None
_IMAGE_EMBED_CACHE_DATA = None

IMAGE_SOURCE_MARKER = "[image]"
IMAGE_URL_RE = re.compile(r"\burl:\s*(https?://\S+)", re.IGNORECASE)
IMAGE_QUERY_HINTS = {
    "rasm",
    "image",
    "belgi",
    "belgilar",
    "piktogramma",
    "piktogrammalar",
    "ikonka",
    "icon",
    "sxema",
    "diagramma",
}
FIGURE_NUMBER_RE = re.compile(
    r"(?:\b(\d+)\s*-\s*rasm(?:ga|da|dan|ni|ning|lar|larga|larda|lardan)?\b|"
    r"\b(\d+)\s*rasm(?:ga|da|dan|ni|ning|lar|larga|larda|lardan)?\b)",
    re.IGNORECASE,
)
FIGURE_PREFIX_RE = re.compile(r"\b(\d+)\s*-\s*(?=,|\s*va\s+\d+\s*-|\s*hamda\s+\d+\s*-)", re.IGNORECASE)
RAG_IMAGE_APPENDIX_TOP_K = int(os.getenv("RAG_IMAGE_APPENDIX_TOP_K", "12"))


def _normalize_text(text: str) -> str:
    lowered = unicodedata.normalize("NFKC", (text or "")).strip().lower()
    lowered = (
        lowered.replace("ʻ", "'")
        .replace("ʼ", "'")
        .replace("‘", "'")
        .replace("’", "'")
        .replace("`", "'")
    )
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _is_image_clause_text(text: str) -> bool:
    return _normalize_text(text).startswith(IMAGE_SOURCE_MARKER)


def _extract_image_url_from_text(text: str):
    if not text:
        return None
    match = IMAGE_URL_RE.search(text)
    if not match:
        return None
    return match.group(1).rstrip(").,;")


def _extract_figure_numbers(text: str):
    value = _normalize_text(text)
    if not value:
        return []
    numbers = []
    for match in FIGURE_PREFIX_RE.finditer(value):
        try:
            numbers.append(int(match.group(1)))
        except Exception:
            continue
    for match in FIGURE_NUMBER_RE.finditer(value):
        raw = match.group(1) or match.group(2)
        if not raw:
            continue
        try:
            numbers.append(int(raw))
        except Exception:
            continue
    unique = []
    seen = set()
    for num in numbers:
        if num in seen:
            continue
        seen.add(num)
        unique.append(num)
    return unique


def _format_clause_text_for_context(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return value
    if not _is_image_clause_text(value):
        return value
    formatted = value.replace("[IMAGE]", "Rasm:")
    return re.sub(r"\s*\|\s*", " | ", formatted)


def _image_text_for_context(image) -> str:
    parts = []
    if image.title:
        parts.append(image.title)
    if image.context_text:
        parts.append(image.context_text)
    if image.ocr_text:
        parts.append(image.ocr_text)
    if image.appendix_number:
        parts.append(f"{image.appendix_number}-ilova")
    parts.append(f"URL: {image.image_url}")
    return " | ".join([p for p in parts if p])


def _query_terms_indicate_image(terms) -> bool:
    if not terms:
        return False
    for term in terms:
        for hint in IMAGE_QUERY_HINTS:
            if term == hint or term.startswith(hint):
                return True
    return False


def _message_indicates_image(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(hint in normalized for hint in IMAGE_QUERY_HINTS)


def _text_term_overlap_score(text: str, terms) -> float:
    if not terms:
        return 0.0
    haystack = _normalize_text(text)
    if not haystack:
        return 0.0
    hits = sum(1 for term in terms if term in haystack)
    return hits / max(len(terms), 1)


def _best_image_segment_match_score(emb: ImageEmbedding, normalized_message: str, terms) -> float:
    image = emb.image
    segments = []
    for value in [image.title, image.section_title, image.context_text, image.ocr_text]:
        if not value:
            continue
        parts = [p.strip() for p in str(value).split("|") if p.strip()]
        if not parts:
            parts = [str(value)]
        segments.extend(parts)
    best = 0.0
    for seg in segments:
        norm_seg = _normalize_text(seg)
        if not norm_seg:
            continue
        score = _text_term_overlap_score(norm_seg, terms)
        if normalized_message and len(normalized_message) >= 12:
            if normalized_message in norm_seg or norm_seg in normalized_message:
                score = max(score, 1.0)
        if score > best:
            best = score
    return best


def _is_greeting(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in GREETING_PATTERNS)


def _is_shnq_related(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(keyword in normalized for keyword in SHNQ_KEYWORDS)


def _is_clearly_out_of_scope(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(keyword in normalized for keyword in OUT_OF_SCOPE_KEYWORDS)


def _build_greeting_response() -> str:
    return "Assalomu alaykum! :) SHNQ bo'yicha qanday savolingiz bor?"


def _build_out_of_scope_response() -> str:
    return (
        "Kechirasiz, men faqat SHNQ (qurilish me'yorlari) bo'yicha savollarga javob bera olaman. "
        "Iltimos, savolingizni SHNQ hujjati, bob yoki bandga bog'lab yozing."
    )


def _contains_any(text: str, keywords) -> bool:
    return any(keyword in text for keyword in keywords)


def _needs_clarification(text: str):
    normalized = _normalize_text(text)
    if not normalized:
        return None

    is_context_sensitive = _contains_any(normalized, CONTEXT_SENSITIVE_TERMS)
    has_use_case = _contains_any(normalized, USE_CASE_KEYWORDS)
    if is_context_sensitive and not has_use_case:
        return CLARIFICATION_RULES[12]

    # Quyidagilar niyatga bog'liq holatlar bo'lib, kerak bo'lganda so'raladi.
    needs_parameter = bool(re.search(r"\b(qancha|necha|minimal|maksimal|me'yor|meyor)\b", normalized))
    has_parameter = bool(re.search(r"\d", normalized)) or _contains_any(normalized, TECH_PARAM_KEYWORDS)
    if needs_parameter and not has_parameter:
        return CLARIFICATION_RULES[6]

    asks_comparison = _contains_any(normalized, COMPARISON_KEYWORDS)
    has_two_targets = normalized.count(" va ") >= 1 or normalized.count(",") >= 1
    if asks_comparison and not has_two_targets:
        return CLARIFICATION_RULES[7]

    mentions_exception_topic = "istisno" in normalized or "maxsus" in normalized
    if mentions_exception_topic and not _contains_any(normalized, EXCEPTION_KEYWORDS):
        return CLARIFICATION_RULES[8]

    asks_explain_only = _contains_any(normalized, REFERENCE_NEED_HINTS)
    has_reference_mode = _contains_any(normalized, REFERENCE_SPECIFIED_HINTS)
    if asks_explain_only and not has_reference_mode:
        return CLARIFICATION_RULES[9]

    if "tushuntir" in normalized and not _contains_any(normalized, EXAMPLE_HINTS):
        return CLARIFICATION_RULES[10]

    broad_request = bool(re.search(r"\b(talab|norma|qoidalar|qanday)\b", normalized))
    if broad_request and not _contains_any(normalized, PURPOSE_HINTS):
        return CLARIFICATION_RULES[11]

    return None


def _is_table_request(text: str) -> bool:
    normalized = _normalize_text(text)
    has_doc = _extract_doc_code(normalized) is not None
    has_table_num = _extract_table_number(normalized) is not None
    has_appendix_num = _extract_appendix_number(normalized) is not None
    has_table_word = bool(TABLE_TYPO_RE.search(normalized))
    return (
        (has_doc and (has_table_num or has_appendix_num))
        or (has_table_num and has_table_word)
        or (has_appendix_num and has_table_word)
        or _contains_any(normalized, TABLE_HINTS)
    )


def _extract_table_number(text: str):
    normalized = _normalize_text(text)
    match = TABLE_NUMBER_RE.search(normalized)
    if not match:
        match = TABLE_NUM_ONLY_RE.search(normalized)
    if not match:
        return None
    return match.group(1) or match.group(2)


def _extract_appendix_number(text: str):
    normalized = _normalize_text(text)
    match = APPENDIX_NUMBER_RE.search(normalized)
    if not match:
        return None
    return match.group(1) or match.group(2)


def _extract_doc_code(text: str):
    match = DOC_CODE_RE.search(_normalize_text(text))
    if not match:
        return None
    prefix = match.group(1).upper()
    number = match.group(2)
    return f"{prefix} {number}"


def _normalize_doc_code(text: str) -> str:
    return re.sub(r"\s+", "", (text or "")).lower()


def _filter_embeddings_by_doc_code(embeddings, doc_code: str):
    target = _normalize_doc_code(doc_code)
    filtered = []
    for emb in embeddings:
        current = _normalize_doc_code(emb.shnq_code or "")
        if not current:
            continue
        if current == target or current.startswith(target):
            filtered.append(emb)
    return filtered


def _table_candidate_docs(table_number: str):
    qs = (
        NormTable.objects.filter(table_number__iexact=table_number)
        .values_list("document__code", flat=True)
        .distinct()
    )
    return list(qs[:7])


def _extract_table_context_terms(message: str, doc_code: str | None, table_number: str | None):
    normalized = _normalize_text(message)
    # Uzbek lotinida apostrofli tokenlarni ham ushlaymiz: ko'p, yo'l, bo'lim va h.k.
    terms = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", normalized, flags=re.UNICODE)
    cleaned = []
    doc_terms = set(
        re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", _normalize_text(doc_code or ""), flags=re.UNICODE)
    )
    number = (table_number or "").lower()
    for term in terms:
        if len(term) < 3:
            continue
        if term in TABLE_CONTEXT_STOP_WORDS:
            continue
        if term in doc_terms:
            continue
        if number and term == number:
            continue
        cleaned.append(term)
    # tartibni saqlagan holda uniq
    uniq = []
    seen = set()
    for term in cleaned:
        if term in seen:
            continue
        seen.add(term)
        uniq.append(term)
    return uniq


def _table_exact_section_hit(table: NormTable, normalized_message: str) -> bool:
    section = _normalize_text(table.section_title or "")
    if len(section) < 8:
        return False
    if section in normalized_message:
        return True
    # "2-§. Ko'p kvartirali ..." kabi sarlavhalarda prefiksni olib tashlab tekshiramiz.
    section_core = re.sub(r"^\d+\s*[-.]?\s*(?:§|bob)?\.?\s*", "", section).strip()
    if len(section_core) >= 8 and section_core in normalized_message:
        return True
    return False


def _table_context_score(table: NormTable, context_terms):
    if not context_terms:
        return 0
    haystack = _normalize_text(
        f"{table.document.code} "
        f"{table.section_title or ''} "
        f"{table.chapter.title if table.chapter else ''} "
        f"{table.title or ''} "
        f"{table.markdown[:2500]} "
        f"{table.raw_html[:2500]}"
    )
    return sum(1 for term in context_terms if term in haystack)


def _table_candidate_chapters(candidates):
    chapters = []
    seen = set()
    for item in candidates:
        chapter = item.section_title or (item.chapter.title if item.chapter else "Noma'lum bob")
        chapter = chapter.strip() if chapter else "Noma'lum bob"
        if chapter.lower() in seen:
            continue
        seen.add(chapter.lower())
        chapters.append(chapter)
        if len(chapters) >= 6:
            break
    return chapters


def _find_table_for_query(message: str):
    table_number = _extract_table_number(message)
    appendix_number = _extract_appendix_number(message)
    if not table_number and appendix_number:
        table_number = f"ilova-{appendix_number}"
    doc_code = _extract_doc_code(message)
    if not table_number:
        return None, table_number, doc_code, []

    candidates = NormTable.objects.select_related("document", "chapter").filter(
        table_number__iexact=table_number
    )
    if doc_code:
        target_code = _normalize_doc_code(doc_code)
        candidates = [item for item in candidates if target_code in _normalize_doc_code(item.document.code)]
    else:
        candidates = list(candidates)

    if not candidates and table_number.startswith("ilova-"):
        appendix_number = table_number.split("-", 1)[1]
        key_variants = {f"{appendix_number}-ilova", f"{appendix_number} ilova"}
        fallback_qs = NormTable.objects.select_related("document", "chapter")
        if doc_code:
            target_code = _normalize_doc_code(doc_code)
            fallback_qs = [item for item in fallback_qs if target_code in _normalize_doc_code(item.document.code)]
        else:
            fallback_qs = list(fallback_qs)
        candidates = [
            item
            for item in fallback_qs
            if any(
                key in _normalize_text(
                    f"{item.section_title or ''} {item.title or ''} {item.markdown[:1200]}"
                )
                for key in key_variants
            )
        ]

    if not candidates:
        return None, table_number, doc_code, []

    normalized_message = _normalize_text(message)
    context_terms = _extract_table_context_terms(message, doc_code, table_number)
    if context_terms:
        exact_matches = [
            item for item in candidates if _table_exact_section_hit(item, normalized_message)
        ]
        if exact_matches:
            # Agar foydalanuvchi bo'limni aniq yozgan bo'lsa, shu bo'limdagi jadvalni qaytaramiz.
            # Bir bo'lim ichida bir nechta 1-jadval bo'lsa ham, tartib bo'yicha birinchisini tanlaymiz.
            section_keys = {
                _normalize_text(item.section_title or item.chapter.title if item.chapter else "")
                for item in exact_matches
            }
            if len(section_keys) == 1:
                picked = sorted(exact_matches, key=lambda x: x.order)[0]
                return picked, table_number, doc_code, candidates

        scored = sorted(
            candidates,
            key=lambda item: (
                1 if _table_exact_section_hit(item, normalized_message) else 0,
                _table_context_score(item, context_terms),
                item.order,
            ),
            reverse=True,
        )
        best = scored[0]
        best_exact = _table_exact_section_hit(best, normalized_message)
        second_exact = _table_exact_section_hit(scored[1], normalized_message) if len(scored) > 1 else False
        best_score = _table_context_score(best, context_terms)
        second_score = _table_context_score(scored[1], context_terms) if len(scored) > 1 else -1
        if best_exact and not second_exact:
            return best, table_number, doc_code, candidates
        if best_score > 0 and best_score > second_score:
            return best, table_number, doc_code, candidates
        if len(scored) == 1:
            return scored[0], table_number, doc_code, candidates
        return None, table_number, doc_code, scored

    if len(candidates) == 1:
        return candidates[0], table_number, doc_code, candidates

    return None, table_number, doc_code, candidates


def _build_table_answer(table: NormTable):
    chapter_title = table.section_title or (table.chapter.title if table.chapter else "-")
    table_ref = table.table_number
    if (table_ref or "").lower().startswith("ilova-"):
        table_ref = f"{table_ref.split('-', 1)[1]}-ilova"
    return (
        f"{table.document.code} bo'yicha {table_ref} topildi "
        f"({chapter_title}). Jadval to'liq ko'rinishda pastda keltirildi."
    )


def _is_table_direct_lookup_request(message: str) -> bool:
    normalized = _normalize_text(message)
    if "?" in (message or ""):
        return False
    return not any(hint in normalized for hint in TABLE_QUESTION_HINTS)


def _is_unhelpful_table_answer(answer: str) -> bool:
    normalized = _normalize_text(answer)
    if not normalized:
        return True
    bad_markers = [
        "javob topilmadi",
        "topilmadi",
        "aniq topilmadi",
        "aniqlanmadi",
        "malumot topilmadi",
        "ma'lumot topilmadi",
    ]
    return any(marker in normalized for marker in bad_markers)


def _to_float(value: str):
    cleaned = (value or "").replace(",", ".").strip()
    try:
        return float(cleaned)
    except Exception:
        return None


def _parse_appendix7_ranges(text: str):
    value = _normalize_text(text)
    pattern = re.compile(
        r"(\d+(?:[.,]\d+)?)\s*dan\s*katta\s*(\d+(?:[.,]\d+)?)\s*gacha|(\d+(?:[.,]\d+)?)\s*gacha",
        re.IGNORECASE,
    )
    ranges = []
    for match in pattern.finditer(value):
        if match.group(1) and match.group(2):
            lo = _to_float(match.group(1))
            hi = _to_float(match.group(2))
            if hi is not None:
                ranges.append((lo, hi))
            continue
        hi = _to_float(match.group(3))
        if hi is not None:
            ranges.append((None, hi))
    return ranges


def _parse_numeric_list(text: str):
    nums = re.findall(r"\d+(?:[.,]\d+)?", text or "")
    return [_to_float(n) for n in nums if _to_float(n) is not None]


def _pick_appendix7_column(message: str):
    normalized = _normalize_text(message)
    if (
        "qurilayotgan bino" in normalized
        or "inshoot yaqinida" in normalized
        or "perimetr" in normalized
        or "tashqarisida" in normalized
    ):
        return 3
    return 2


def _table_local_numeric_answer(message: str, table: NormTable):
    is_appendix7 = (table.table_number or "").lower() == "ilova-7" or "xavfli zonalarning chegaralari" in _normalize_text(
        f"{table.title or ''} {table.section_title or ''}"
    )
    if not is_appendix7:
        return None

    normalized_message = _normalize_text(message)
    normalized_message = re.sub(
        r"\b\d+\s*[-.]?\s*ilova(?:si|da|ga|dan|ning|lar)?\b",
        " ",
        normalized_message,
        flags=re.IGNORECASE,
    )
    normalized_message = re.sub(
        r"\bilova(?:si|da|ga|dan|ning|lar)?\s*[-.]?\s*\d+\b",
        " ",
        normalized_message,
        flags=re.IGNORECASE,
    )
    query_num_match = re.search(r"\d+(?:[.,]\d+)?", normalized_message)
    if not query_num_match:
        return None
    target_value = _to_float(query_num_match.group(0))
    if target_value is None:
        return None

    rows_data = []
    for row in table.rows.prefetch_related("cells").all():
        cell_map = {}
        for cell in row.cells.all():
            cell_map[cell.col_index] = (cell.text or "").strip()
        rows_data.append(cell_map)

    candidate = None
    for cell_map in rows_data:
        c1 = cell_map.get(1, "")
        c2 = cell_map.get(2, "")
        c3 = cell_map.get(3, "")
        if "gacha" in _normalize_text(c1) and re.search(r"\d", c2) and re.search(r"\d", c3):
            candidate = cell_map
            break
    if not candidate:
        return None

    ranges = _parse_appendix7_ranges(candidate.get(1, ""))
    vals2 = _parse_numeric_list(candidate.get(2, ""))
    vals3 = _parse_numeric_list(candidate.get(3, ""))
    if not ranges or len(ranges) != len(vals2) or len(ranges) != len(vals3):
        return None

    chosen_idx = None
    for idx, (lo, hi) in enumerate(ranges):
        if hi is None:
            continue
        if target_value <= hi and (lo is None or target_value > lo):
            chosen_idx = idx
            break
    if chosen_idx is None:
        return None

    col = _pick_appendix7_column(message)
    result = vals2[chosen_idx] if col == 2 else vals3[chosen_idx]
    side = (
        "kran yordamida yuk ko'chiriladigan joylar"
        if col == 2
        else "qurilayotgan bino/inshoot perimetridan tashqarisi"
    )
    pretty_target = str(int(target_value)) if float(target_value).is_integer() else str(target_value).replace(".", ",")
    pretty_result = str(int(result)) if float(result).is_integer() else str(result).replace(".", ",")
    return (
        f"Jadvalga ko'ra, buyumning mumkin bo'lgan tushish balandligi {pretty_target} m bo'lsa, "
        f"{side} uchun xavfli zona chegarasi {pretty_result} m."
    )


def _build_table_qa_answer(message: str, table: NormTable) -> str:
    local_answer = _table_local_numeric_answer(message, table)
    if local_answer:
        return local_answer

    system = (
        "Siz SHNQ jadvali bo'yicha yordamchisiz. Faqat berilgan jadval matniga tayangan holda javob bering. "
        "Jadvalda aniq topilmasa, shu holatni ochiq ayting va taxmin qilmang."
    )
    table_ref = table.table_number
    if (table_ref or "").lower().startswith("ilova-"):
        table_ref = f"{table_ref.split('-', 1)[1]}-ilova"
    prompt = (
        f"Savol: {message}\n\n"
        f"Hujjat: {table.document.code}\n"
        f"Bo'lim: {table.section_title or (table.chapter.title if table.chapter else '-')}\n"
        f"Jadval/ilova identifikatori: {table_ref}\n"
        "Jadval (markdown):\n"
        f"{table.markdown}\n\n"
        "Javobni qisqa va aniq yozing, kerak bo'lsa qaysi satr/ustundan olganingizni ayting."
    )
    try:
        answer = generate_text(
            prompt,
            system=system,
            model=CHAT_MODEL,
            options={"temperature": 0.0, "top_p": 0.9, "max_tokens": RAG_TABLE_QA_MAX_TOKENS},
        )
        if answer and not _is_unhelpful_table_answer(answer):
            return answer
    except Exception:
        pass
    return _build_table_answer(table)


def _pick_related_table_from_rag(message: str, top_pairs):
    if not top_pairs:
        return None
    top_clause_text = top_pairs[0][1].clause.text if top_pairs and top_pairs[0][1] else ""

    # Agar savolda jadval ishorasi bo'lmasa, lekin topilgan band ilovaga havola qilsa,
    # so'rovni implicit jadval so'rovi deb ko'rib chiqamiz.
    if not _is_table_request(message):
        implied_appendix = _extract_appendix_number(top_clause_text)
        if not implied_appendix:
            return None
        message = f"{message} {implied_appendix}-ilova jadval"

    table_number = _extract_table_number(message)
    appendix_number = _extract_appendix_number(message) or _extract_appendix_number(top_clause_text)
    if not table_number and appendix_number:
        table_number = f"ilova-{appendix_number}"
    doc_code = _extract_doc_code(message)
    top_doc = top_pairs[0][1].shnq_code if top_pairs and top_pairs[0][1] else None
    target_doc = doc_code or top_doc

    qs = NormTable.objects.select_related("document", "chapter")
    if target_doc:
        target_norm = _normalize_doc_code(target_doc)
        qs = [t for t in qs if target_norm in _normalize_doc_code(t.document.code)]
    else:
        qs = list(qs)
    if table_number:
        filtered = [t for t in qs if (t.table_number or "").lower() == table_number.lower()]
        if filtered:
            qs = filtered
        elif table_number.startswith("ilova-"):
            appendix_number = table_number.split("-", 1)[1]
            key_variants = {f"{appendix_number}-ilova", f"{appendix_number} ilova"}
            qs = [
                t
                for t in qs
                if any(
                    key in _normalize_text(
                        f"{t.section_title or ''} {t.title or ''} {t.markdown[:1200]}"
                    )
                    for key in key_variants
                )
            ]

    if not qs:
        return None

    normalized = _normalize_text(message)
    context_terms = _extract_table_context_terms(message, target_doc, table_number)
    ranked = sorted(
        qs,
        key=lambda t: (
            1 if _table_exact_section_hit(t, normalized) else 0,
            _table_context_score(t, context_terms),
            t.order,
        ),
        reverse=True,
    )
    best = ranked[0]
    best_score = _table_context_score(best, context_terms)
    if table_number:
        return best
    if best_score > 0:
        return best
    return None


def _ensure_embeddings():
    global _EMBED_CACHE_MODEL, _EMBED_CACHE_DATA
    global _IMAGE_EMBED_CACHE_MODEL, _IMAGE_EMBED_CACHE_DATA

    ensure_runtime_tables()

    total_clauses = Clause.objects.count()
    total_images = NormImage.objects.count()
    existing_clauses = ClauseEmbedding.objects.filter(embedding_model=EMBEDDING_MODEL).count()
    existing_images = ImageEmbedding.objects.filter(embedding_model=EMBEDDING_MODEL).count()

    needs_upsert_clauses = total_clauses > 0 and existing_clauses < total_clauses
    needs_upsert_images = total_images > 0 and existing_images < total_images

    needs_upsert = needs_upsert_clauses or needs_upsert_images
    if USE_QDRANT:
        qdrant_total = qdrant_count_points()
        if qdrant_total < existing_clauses:
            needs_upsert = True

    if not needs_upsert:
        return

    upsert_all_embeddings(embedding_model=EMBEDDING_MODEL, force_update=False)
    _EMBED_CACHE_MODEL = None
    _EMBED_CACHE_DATA = None
    _IMAGE_EMBED_CACHE_MODEL = None
    _IMAGE_EMBED_CACHE_DATA = None


def _prepare_embedding_runtime_fields(embeddings):
    for emb in embeddings:
        emb._norm_clause = _normalize_text(emb.clause.text)
        emb._norm_chapter = _normalize_text(emb.chapter_title or "")
        emb._norm_code = _normalize_text(emb.shnq_code or "")
    return embeddings


def _get_embeddings_for_query():
    global _EMBED_CACHE_MODEL, _EMBED_CACHE_DATA

    if EMBED_CACHE_ENABLED and _EMBED_CACHE_MODEL == EMBEDDING_MODEL and _EMBED_CACHE_DATA is not None:
        return _EMBED_CACHE_DATA

    ensure_runtime_tables()
    data = list(
        ClauseEmbedding.objects.select_related("clause", "clause__document", "clause__chapter").filter(
            embedding_model=EMBEDDING_MODEL
        )
    )
    data = _prepare_embedding_runtime_fields(data)
    if EMBED_CACHE_ENABLED:
        _EMBED_CACHE_MODEL = EMBEDDING_MODEL
        _EMBED_CACHE_DATA = data
    return data


def _prepare_image_embedding_runtime_fields(embeddings):
    for emb in embeddings:
        image = emb.image
        emb._norm_text = _normalize_text(
            " ".join(
                [
                    image.title or "",
                    image.context_text or "",
                    image.ocr_text or "",
                    emb.chapter_title or "",
                    emb.shnq_code or "",
                    emb.appendix_number or "",
                ]
            )
        )
        emb._norm_code = _normalize_text(emb.shnq_code or "")
    return embeddings


def _get_image_embeddings_for_query():
    global _IMAGE_EMBED_CACHE_MODEL, _IMAGE_EMBED_CACHE_DATA

    if (
        EMBED_CACHE_ENABLED
        and _IMAGE_EMBED_CACHE_MODEL == EMBEDDING_MODEL
        and _IMAGE_EMBED_CACHE_DATA is not None
    ):
        return _IMAGE_EMBED_CACHE_DATA

    ensure_runtime_tables()
    data = list(
        ImageEmbedding.objects.select_related("image", "image__document", "image__chapter").filter(
            embedding_model=EMBEDDING_MODEL
        )
    )
    data = _prepare_image_embedding_runtime_fields(data)
    if EMBED_CACHE_ENABLED:
        _IMAGE_EMBED_CACHE_MODEL = EMBEDDING_MODEL
        _IMAGE_EMBED_CACHE_DATA = data
    return data


def _score_with_qdrant(query_vec, query_terms, requested_doc_code=None):
    if not query_vec:
        return []
    limit = RAG_QDRANT_DOC_LIMIT if requested_doc_code else RAG_QDRANT_LIMIT
    hits = qdrant_search(query_vec, limit=limit, doc_code=requested_doc_code)
    if not hits:
        return []

    ids = []
    for hit in hits:
        hit_id = str(hit.id)
        if hit_id.startswith("clause:"):
            hit_id = hit_id.split(":", 1)[1]
        ids.append(hit_id)
    embeddings = ClauseEmbedding.objects.select_related("clause", "clause__document", "clause__chapter").filter(
        clause_id__in=ids
    )
    emb_map = {str(emb.clause_id): emb for emb in embeddings}

    scored = []
    for hit in hits:
        hit_id = str(hit.id)
        if hit_id.startswith("clause:"):
            hit_id = hit_id.split(":", 1)[1]
        emb = emb_map.get(hit_id)
        if not emb or hit.score is None:
            continue
        semantic = float(hit.score)
        keyword = _keyword_score(query_terms, emb)
        score = semantic + (KEYWORD_WEIGHT * keyword)
        scored.append((score, emb, semantic, keyword))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _rewrite_query_if_needed(question: str) -> str:
    if not REWRITE_QUERY:
        return question
    system = (
        "Siz SHNQ qidiruv yordamchisisiz. Savolni qisqa va aniq qidiruv so'roviga aylantiring. "
        "Faqat bitta qatorda qaytaring. Yangi fakt qo'shmang."
    )
    prompt = (
        "Quyidagi savolni SHNQ hujjatlari bo'yicha qidiruvga moslab qayta yozing:\n"
        f"{question}\n\nQidiruv so'rovi:"
    )
    try:
        rewritten = generate_text(
            prompt,
            system=system,
            model=CHAT_MODEL,
            options={"temperature": 0.0, "top_p": 0.9, "max_tokens": RAG_REWRITE_MAX_TOKENS},
        )
        return rewritten or question
    except Exception:
        return question


def _extract_query_terms(text: str):
    normalized = _normalize_text(text)
    # Unicode harf/raqam tokenlarini ajratamiz (uz/en/ru/ko va boshqalar uchun).
    terms = re.findall(r"[^\W_]+(?:'[^\W_]+)?", normalized, flags=re.UNICODE)
    # Juda qisqa tokenlarni olib tashlaymiz.
    terms = [t for t in terms if len(t) >= 3]
    # Stop-so'zlarni kamaytiramiz.
    stop_words = {
        "uchun",
        "bilan",
        "qanday",
        "nima",
        "kerak",
        "emas",
        "bolsa",
        "bormi",
        "haqida",
        "boyicha",
        "qaysi",
        "qilib",
        "buni",
    }
    filtered = [t for t in terms if t not in stop_words]
    # Takrorlarni olib tashlab, tartibni saqlaymiz.
    seen = set()
    unique_terms = []
    for term in filtered:
        if term in seen:
            continue
        seen.add(term)
        unique_terms.append(term)
        if len(unique_terms) >= MAX_QUERY_TERMS:
            break
    return unique_terms


def _keyword_score(terms, emb: ClauseEmbedding) -> float:
    if not terms:
        return 0.0
    clause_text = getattr(emb, "_norm_clause", None) or _normalize_text(emb.clause.text)
    chapter = getattr(emb, "_norm_chapter", None) or _normalize_text(emb.chapter_title or "")
    shnq_code = getattr(emb, "_norm_code", None) or _normalize_text(emb.shnq_code or "")
    is_image_clause = _is_image_clause_text(emb.clause.text or "")
    asks_image = _query_terms_indicate_image(terms)
    if is_image_clause and not asks_image:
        # Rasm bloklari oddiy matnli norma qidiruvini siqib chiqarmasin.
        return 0.0
    hits = 0
    for term in terms:
        if term in clause_text:
            hits += 1
            continue
        if term in chapter or term in shnq_code:
            hits += 1
    score = hits / max(len(terms), 1)
    if is_image_clause and asks_image:
        score = min(1.0, score + 0.25)
    return score


def _image_keyword_score(terms, emb: ImageEmbedding) -> float:
    if not terms:
        return 0.0
    haystack = getattr(emb, "_norm_text", None) or _normalize_text(
        " ".join(
            [
                emb.image.title or "",
                emb.image.context_text or "",
                emb.image.ocr_text or "",
                emb.chapter_title or "",
                emb.shnq_code or "",
                emb.appendix_number or "",
            ]
        )
    )
    hits = sum(1 for term in terms if term in haystack)
    return hits / max(len(terms), 1)


def _filter_image_embeddings_by_doc_code(embeddings, doc_code: str):
    target = _normalize_doc_code(doc_code)
    filtered = []
    for emb in embeddings:
        current = _normalize_doc_code(emb.shnq_code or "")
        if not current:
            continue
        if current == target or current.startswith(target):
            filtered.append(emb)
    return filtered


def _search_image_embeddings(message: str, requested_doc_code: str | None = None, limit: int | None = None):
    terms = _extract_query_terms(message)
    if not (_query_terms_indicate_image(terms) or _message_indicates_image(message)):
        return []
    normalized_message = _normalize_text(message)

    # Agar foydalanuvchi matni rasm kontekstida aniq ibora sifatida uchrasa,
    # faqat shu(lar)ni qaytaramiz (masalan: aniq belgi nomi/sarlavha).
    if len(normalized_message) >= 12:
        phrase_matched = []
        base_embeddings = _get_image_embeddings_for_query()
        if requested_doc_code:
            base_embeddings = _filter_image_embeddings_by_doc_code(base_embeddings, requested_doc_code)
        for emb in base_embeddings:
            haystack = _normalize_text(
                " ".join(
                    [
                        emb.image.title or "",
                        emb.image.section_title or "",
                        emb.image.context_text or "",
                        emb.image.ocr_text or "",
                    ]
                )
            )
            if normalized_message in haystack:
                phrase_matched.append(emb)
        if phrase_matched:
            phrase_ranked = sorted(
                phrase_matched,
                key=lambda e: _text_term_overlap_score(
                    f"{e.image.title or ''} {e.image.section_title or ''} {e.image.context_text or ''} {e.image.ocr_text or ''}",
                    terms,
                ),
                reverse=True,
            )
            max_items = limit or RAG_IMAGE_TOP_K
            return [(1.0 - (idx * 0.001), emb, 1.0, 1.0) for idx, emb in enumerate(phrase_ranked[:max_items])]

        # Foydalanuvchi aniq sarlavha/bo'lak so'ragan bo'lsa, bitta eng mos rasmni tanlaymiz.
        segment_ranked = sorted(
            [
                (
                    _best_image_segment_match_score(emb, normalized_message, terms),
                    emb,
                )
                for emb in base_embeddings
            ],
            key=lambda x: x[0],
            reverse=True,
        )
        if segment_ranked and segment_ranked[0][0] >= 0.72:
            second = segment_ranked[1][0] if len(segment_ranked) > 1 else 0.0
            if (segment_ranked[0][0] - second) >= 0.12:
                best_emb = segment_ranked[0][1]
                return [(1.0, best_emb, 1.0, 1.0)]
    try:
        query_vec = embed_text(message, model=EMBEDDING_MODEL)
    except Exception:
        return []
    if not query_vec:
        return []

    embeddings = _get_image_embeddings_for_query()
    if requested_doc_code:
        embeddings = _filter_image_embeddings_by_doc_code(embeddings, requested_doc_code)

    scored = []
    for emb in embeddings:
        if not emb.vector:
            continue
        semantic = cosine_similarity(query_vec, emb.vector)
        keyword = _image_keyword_score(terms, emb)
        score = semantic + (KEYWORD_WEIGHT * keyword)
        if score < RAG_IMAGE_MIN_SCORE:
            continue
        scored.append((score, emb, semantic, keyword))

    scored.sort(key=lambda x: x[0], reverse=True)
    max_items = limit or RAG_IMAGE_TOP_K
    if scored:
        return scored[: max_items]

    # Embedding skori past bo'lib qolgan hollarda matn mosligi bo'yicha fallback.
    fallback = []
    for emb in embeddings:
        keyword = _image_keyword_score(terms, emb)
        if keyword <= 0:
            continue
        hit_count = int(round(keyword * max(len(terms), 1)))
        if hit_count < 2 and keyword < 0.25:
            continue
        score = KEYWORD_WEIGHT * keyword
        fallback.append((score, emb, 0.0, keyword))

    fallback.sort(key=lambda x: (x[3], x[0]), reverse=True)
    return fallback[: max_items]


def _linked_image_embeddings_from_clause_refs(top_pairs, message: str, requested_doc_code: str | None = None):
    if not top_pairs:
        return []

    target_doc = requested_doc_code or (top_pairs[0][1].shnq_code if top_pairs and top_pairs[0][1] else None)
    if not target_doc:
        return []
    target_doc_norm = _normalize_doc_code(target_doc)

    appendix_number = _extract_appendix_number(message)
    query_figure_numbers = _extract_figure_numbers(message)
    figure_numbers = list(query_figure_numbers)
    query_terms = _extract_query_terms(message)

    for _score, emb, _semantic, _keyword in top_pairs[:3]:
        clause_text = emb.clause.text or ""
        if not appendix_number:
            appendix_number = _extract_appendix_number(clause_text)
        figure_numbers.extend(_extract_figure_numbers(clause_text))

    unique_figure_numbers = []
    seen_figures = set()
    for num in figure_numbers:
        if num in seen_figures:
            continue
        seen_figures.add(num)
        unique_figure_numbers.append(num)

    if not appendix_number and not unique_figure_numbers:
        return []

    image_candidates = [
        image
        for image in NormImage.objects.select_related("document").order_by("order")
        if _normalize_doc_code(image.document.code).startswith(target_doc_norm)
    ]
    if appendix_number:
        appendix_str = str(appendix_number)
        image_candidates = [img for img in image_candidates if str(img.appendix_number or "") == appendix_str]
    if not image_candidates:
        return []

    picked_images = []
    full_appendix_requested = False
    if appendix_number and not query_figure_numbers and query_terms:
        # Savol sarlavha/bo'limga qaratilgan bo'lsa (masalan: "Himoya to'rlarining turlari"),
        # ilovadagi barcha rasmlarni qaytaramiz.
        heading_score = _text_term_overlap_score(
            " ".join(
                [
                    image_candidates[0].title or "",
                    image_candidates[0].section_title or "",
                    image_candidates[0].context_text or "",
                ]
            ),
            query_terms,
        )
        if heading_score >= 0.34 or "turlari" in _normalize_text(message):
            full_appendix_requested = True

    if full_appendix_requested:
        picked_images = image_candidates[:RAG_IMAGE_APPENDIX_TOP_K]
    elif appendix_number and unique_figure_numbers:
        for num in unique_figure_numbers:
            idx = num - 1
            if 0 <= idx < len(image_candidates):
                picked_images.append(image_candidates[idx])

    if not picked_images and unique_figure_numbers:
        for img in image_candidates:
            haystack = _normalize_text(" ".join([img.title or "", img.context_text or "", img.ocr_text or ""]))
            if any(f"{num}-rasm" in haystack or f"{num} rasm" in haystack for num in unique_figure_numbers):
                picked_images.append(img)

    if not picked_images:
        return []

    # first-match tartibida uniq va limit
    uniq_images = []
    seen_ids = set()
    for img in picked_images:
        if str(img.id) in seen_ids:
            continue
        seen_ids.add(str(img.id))
        uniq_images.append(img)
        max_pick = RAG_IMAGE_APPENDIX_TOP_K if full_appendix_requested else RAG_IMAGE_TOP_K
        if len(uniq_images) >= max_pick:
            break
    if not uniq_images:
        return []

    emb_qs = ImageEmbedding.objects.select_related("image", "image__document", "image__chapter").filter(
        image_id__in=[img.id for img in uniq_images],
        embedding_model=EMBEDDING_MODEL,
    )
    emb_map = {str(item.image_id): item for item in emb_qs}

    base_score = top_pairs[0][0] if top_pairs else RAG_IMAGE_MIN_SCORE
    results = []
    for idx, img in enumerate(uniq_images):
        emb = emb_map.get(str(img.id))
        if not emb:
            continue
        linked_score = max(RAG_IMAGE_MIN_SCORE, min(1.0, base_score - (idx * 0.005)))
        results.append((linked_score, emb, linked_score, 1.0))
    return results


def _candidate_documents_from_scored(scored, best_score):
    if not scored:
        return []
    threshold = max(RAG_LOW_CONFIDENCE_FLOOR, best_score - RAG_AMBIGUITY_SCORE_GAP)
    docs = []
    seen = set()
    for score, emb, _semantic, _keyword in scored:
        if score < threshold:
            break
        code = (emb.shnq_code or "").strip()
        key = code.lower()
        if not code or key in seen:
            continue
        seen.add(key)
        docs.append(code)
        if len(docs) >= RAG_AMBIGUITY_MAX_DOCS:
            break
    return docs


def _should_ask_document_clarification(scored, best_score):
    if len(scored) < 2:
        return False, []
    docs = _candidate_documents_from_scored(scored, best_score)
    if len(docs) <= 1:
        return False, docs

    second_score = scored[1][0]
    close_scores = (best_score - second_score) <= RAG_AMBIGUITY_SCORE_GAP
    low_confidence = best_score < STRICT_MIN_SCORE
    many_variants = len(docs) >= 3 and best_score < (STRICT_MIN_SCORE + 0.05)
    return close_scores or low_confidence or many_variants, docs


def _build_document_clarification_answer(docs):
    return (
        "Savolda bir nechta hujjatda mos variant topildi. "
        f"Aniq javob uchun qaysi hujjat kerakligini tanlang: {', '.join(docs)}."
    )


def _dominant_doc_ratio(scored):
    if not scored:
        return 0.0
    window = scored[: max(1, RAG_DOMINANCE_WINDOW)]
    counts = {}
    for _score, emb, _semantic, _keyword in window:
        code = (emb.shnq_code or "").strip().lower()
        if not code:
            continue
        counts[code] = counts.get(code, 0) + 1
    if not counts:
        return 0.0
    return max(counts.values()) / max(len(window), 1)


def _can_answer_with_relaxed_threshold(scored, best_score):
    if not scored:
        return False
    if best_score >= STRICT_MIN_SCORE:
        return True
    if best_score < max(MIN_SCORE, STRICT_MIN_SCORE - RAG_NEAR_STRICT_MARGIN):
        return False

    dominant_ratio = _dominant_doc_ratio(scored)
    top_keyword_score = scored[0][3]
    return dominant_ratio >= RAG_DOC_DOMINANCE_MIN_RATIO and top_keyword_score >= RAG_STRONG_KEYWORD_MIN


def _llm_rerank(question: str, candidates):
    if not RERANK_ENABLED or not candidates:
        return candidates
    lines = []
    for idx, (score, emb, semantic, keyword) in enumerate(candidates, 1):
        snippet = (emb.clause.text or "")[:240].replace("\n", " ")
        lines.append(
            f"{idx}) {emb.shnq_code} | bob: {emb.chapter_title or '-'} | band: {emb.clause_number or '-'} | {snippet}"
        )
    system = (
        "Siz SHNQ bo'yicha relevans reytingchisiz. "
        "Berilgan savolga eng mos bandlarni tanlang."
    )
    prompt = (
        f"Savol: {question}\n\n"
        "Quyidagi variantlardan eng mos 5 tasining raqamini tanlang.\n"
        "Faqat raqamlar ro'yxatini qaytaring, masalan: 3,1,5,2,4\n\n"
        "Variantlar:\n"
        + "\n".join(lines)
        + "\n\nTanlangan raqamlar:"
    )
    try:
        raw = generate_text(
            prompt,
            system=system,
            model=CHAT_MODEL,
            options={"temperature": 0.0, "top_p": 0.9, "max_tokens": RAG_RERANK_MAX_TOKENS},
        )
    except Exception:
        return candidates

    if not raw:
        return candidates

    # Raqamlarni xavfsiz pars qilamiz.
    picked = []
    for token in re.findall(r"\d+", raw):
        i = int(token)
        if 1 <= i <= len(candidates) and i not in picked:
            picked.append(i)
        if len(picked) >= 5:
            break

    if not picked:
        return candidates

    # Tanlanganlarni oldinga, qolganlarni o'z tartibida qoldiramiz.
    picked_set = set(picked)
    ordered = [candidates[i - 1] for i in picked]
    ordered.extend(c for idx, c in enumerate(candidates, 1) if idx not in picked_set)
    return ordered


_FEWSHOT_CACHE = None
_FEWSHOT_VECTOR_CACHE = {}


def _load_fewshot_examples():
    global _FEWSHOT_CACHE
    if _FEWSHOT_CACHE is not None:
        return _FEWSHOT_CACHE

    if os.getenv("RAG_FEWSHOT_ENABLED", "1") != "1":
        _FEWSHOT_CACHE = []
        return _FEWSHOT_CACHE

    file_path = Path(os.getenv("RAG_FEWSHOT_FILE", "app_shnq/data/qa_fewshot.json"))
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    if not file_path.exists():
        _FEWSHOT_CACHE = []
        return _FEWSHOT_CACHE

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        _FEWSHOT_CACHE = []
        return _FEWSHOT_CACHE

    cleaned = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        if not question or not answer:
            continue
        cleaned.append({"question": question, "answer": answer})

    _FEWSHOT_CACHE = cleaned
    return _FEWSHOT_CACHE


def _pick_fewshot_examples(question: str, limit: int = 3):
    examples = _load_fewshot_examples()
    if not examples:
        return []

    try:
        query_vec = embed_text(question, model=EMBEDDING_MODEL)
    except Exception:
        query_vec = None
    if not query_vec:
        return examples[: max(limit, 1)]

    ranked = []
    for item in examples:
        q_text = item["question"]
        vec = _FEWSHOT_VECTOR_CACHE.get(q_text)
        if vec is None:
            try:
                vec = embed_text(q_text, model=EMBEDDING_MODEL)
            except Exception:
                vec = None
            _FEWSHOT_VECTOR_CACHE[q_text] = vec
        if not vec:
            continue
        ranked.append((cosine_similarity(query_vec, vec), item))

    if not ranked:
        return examples[: max(limit, 1)]
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked[: max(limit, 1)]]


def _build_rag_prompt(question, sources, response_language="uz", fewshot_examples=None, image_sources=None):
    context_chunks = []
    for idx, emb in enumerate(sources, 1):
        clause = emb.clause
        header = f"Manba {idx}"
        lines = [
            header,
            f"Hujjat: {emb.shnq_code}",
            f"Bob: {emb.chapter_title or 'Nomalum bob'}",
            f"Band: {emb.clause_number or '-'}",
            f"Matn: {_format_clause_text_for_context(clause.text)}",
        ]
        context_chunks.append("\n".join(lines))

    if image_sources:
        base_idx = len(context_chunks)
        for offset, image_emb in enumerate(image_sources, 1):
            image = image_emb.image
            header = f"Manba {base_idx + offset} (Rasm)"
            lines = [
                header,
                f"Hujjat: {image_emb.shnq_code}",
                f"Bob: {image_emb.chapter_title or 'Nomalum bob'}",
                f"Ilova: {image_emb.appendix_number or '-'}",
                f"Rasm URL: {image.image_url}",
                f"Matn: {_image_text_for_context(image)}",
            ]
            context_chunks.append("\n".join(lines))

    context = "\n\n".join(context_chunks)
    fewshot_block = ""
    if fewshot_examples:
        samples = []
        for idx, item in enumerate(fewshot_examples, 1):
            samples.append(
                f"Namuna {idx}\nSavol: {item['question']}\nJavob: {item['answer']}"
            )
        fewshot_block = "\n\nJavob uslubi namunalari:\n" + "\n\n".join(samples)
    language_label = {"uz": "o'zbek", "en": "ingliz", "ru": "rus", "ko": "koreys"}.get(response_language, "o'zbek")
    system = (
        "Siz SHNQ AI'siz. Faqat SHNQ/QMQ va qurilish normalari hujjatlariga tayangan holda javob bering. "
        "Hech qachon normani o'ylab topmang, talqin qilmang, faqat kontekstdagi faktlarni yozing. "
        "Kontekstda javob bo'lmasa, buni ochiq ayting. "
        f"Javobni {language_label} tilida yozing. "
        "Javob formatida raqamli punktlar ((1), (2), (3), (4)) ishlatmang. "
        "Javobni 2 qismda bering: avval 'Batafsil:' deb mazmunli va to'liq tushuntiring, "
        "so'ng 'Qisqa qilib aytganda:' deb 1-2 jumlada xulosa bering."
    )
    prompt = f"Savol: {question}\n\nKontekst:\n{context}{fewshot_block}\n\nJavob:"
    return system, prompt


def _cleanup_answer_format(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return text

    # LLM eski promptga ko'ra (1)..(4) qaytarsa, 1-2 ni olib tashlab, 3-4 ni kerakli ko'rinishga o'tkazamiz.
    text = re.sub(r"\(\s*1\s*\)\s*[^.\n:]*[:.]?\s*.*?(?:\n|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*2\s*\)\s*[^.\n:]*[:.]?\s*.*?(?:\n|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*3\s*\)\s*[^.\n:]*[:.]?\s*", "Batafsil: ", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*4\s*\)\s*[^.\n:]*[:.]?\s*", "\nQisqa qilib aytganda: ", text, flags=re.IGNORECASE)
    # Javob matnida manba qatori ko'rsatilmasin (manba alohida `sources` da mavjud).
    text = re.sub(r"^\s*manba\s*:\s*.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if "Batafsil:" not in text:
        text = f"Batafsil: {text}"
    if "Qisqa qilib aytganda:" not in text:
        # Qisqa xulosani model bermagan bo'lsa, mavjud matndan ixcham yakun qo'shamiz.
        short = text.replace("Batafsil:", "").strip()
        # Band raqami prefiksini olib tashlaymiz: "23. ..." -> "..."
        short = re.sub(r"^\s*\d+\s*[.)-]\s*", "", short)
        short = short.split(".")[0].strip()
        if len(short.split()) > 18:
            short = " ".join(short.split()[:18]).strip() + "..."
        if short:
            text = f"{text}\nQisqa qilib aytganda: {short}."
    return text



