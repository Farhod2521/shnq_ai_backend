import logging
import os
import re
import time
import unicodedata

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .deepseek_client import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    detect_query_language,
    embed_text,
    ensure_answer_language,
    generate_text,
    translate_query_for_search,
    translate_html_preserving_tags,
)
from .embeddings import cosine_similarity, upsert_clause_embeddings
from .models import Clause, ClauseEmbedding, NormTable, QuestionAnswer
from .qdrant_store import count_points as qdrant_count_points, search as qdrant_search


EMBEDDING_MODEL = os.getenv("DEEPSEEK_EMBED_MODEL", DEFAULT_EMBED_MODEL)
CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", DEFAULT_CHAT_MODEL)
RAG_FAST_MODE = os.getenv("RAG_FAST_MODE", "1") == "1"
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.2"))
STRICT_MIN_SCORE = float(os.getenv("RAG_STRICT_MIN_SCORE", "0.35"))
KEYWORD_WEIGHT = float(os.getenv("RAG_KEYWORD_WEIGHT", "0.15"))
MAX_QUERY_TERMS = int(os.getenv("RAG_MAX_QUERY_TERMS", "8"))
REWRITE_QUERY = os.getenv("RAG_REWRITE_QUERY", "0") == "1"
RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "0") == "1"
RERANK_CANDIDATES = int(os.getenv("RAG_RERANK_CANDIDATES", "12"))
EMBED_CACHE_ENABLED = os.getenv("RAG_EMBED_CACHE", "1") == "1"
RAG_FINAL_MAX_TOKENS = int(os.getenv("RAG_FINAL_MAX_TOKENS", "220" if RAG_FAST_MODE else "420"))
RAG_REWRITE_MAX_TOKENS = int(os.getenv("RAG_REWRITE_MAX_TOKENS", "80"))
RAG_RERANK_MAX_TOKENS = int(os.getenv("RAG_RERANK_MAX_TOKENS", "40"))
RAG_TABLE_QA_MAX_TOKENS = int(os.getenv("RAG_TABLE_QA_MAX_TOKENS", "180" if RAG_FAST_MODE else "280"))
RAG_AMBIGUITY_SCORE_GAP = float(os.getenv("RAG_AMBIGUITY_SCORE_GAP", "0.03"))
RAG_AMBIGUITY_MAX_DOCS = int(os.getenv("RAG_AMBIGUITY_MAX_DOCS", "6"))
RAG_LOW_CONFIDENCE_FLOOR = float(os.getenv("RAG_LOW_CONFIDENCE_FLOOR", "0.12"))
RAG_NEAR_STRICT_MARGIN = float(os.getenv("RAG_NEAR_STRICT_MARGIN", "0.03"))
RAG_DOC_DOMINANCE_MIN_RATIO = float(os.getenv("RAG_DOC_DOMINANCE_MIN_RATIO", "0.8"))
RAG_STRONG_KEYWORD_MIN = float(os.getenv("RAG_STRONG_KEYWORD_MIN", "0.3"))
RAG_DOMINANCE_WINDOW = int(os.getenv("RAG_DOMINANCE_WINDOW", "5"))
USE_QDRANT = os.getenv("RAG_USE_QDRANT", "0") == "1"
RAG_QDRANT_LIMIT = int(os.getenv("RAG_QDRANT_LIMIT", "60"))
RAG_QDRANT_DOC_LIMIT = int(os.getenv("RAG_QDRANT_DOC_LIMIT", "200"))

_EMBED_CACHE_MODEL = None
_EMBED_CACHE_DATA = None
logger = logging.getLogger(__name__)

GREETING_PATTERNS = [
    r"\bsalom\b",
    r"\bassalomu?\s+alaykum\b",
    r"\bva\s*alaykum\s+assalom\b",
    r"\bhello\b",
    r"\bhi\b",
    r"\bhayrli\s+kun\b",
    r"\bhayrli\s+tong\b",
    r"\bhayrli\s+kech\b",
]

SHNQ_KEYWORDS = [
    "shnq",
    "qurilish",
    "me'yor",
    "meyor",
    "me'yoriy",
    "norma",
    "normalar",
    "standart",
    "band",
    "bob",
    "hujjat",
    "smeta",
    "loyiha",
    "qmq",
    "snip",
    "kmk",
]

OUT_OF_SCOPE_KEYWORDS = [
    "python",
    "javascript",
    "js",
    "react",
    "nextjs",
    "next.js",
    "fastapi",
    "django",
    "sql",
    "kod",
    "code",
    "program",
    "dastur",
    "ob-havo",
    "ob havo",
    "weather",
    "sport",
    "futbol",
    "music",
    "kino",
    "tarjima",
    "translate",
]

DOCUMENT_CODE_RE = re.compile(r"\b(?:shnq|qmq|kmk|snip)\s*\d", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

OBJECT_TYPE_KEYWORDS = [
    "turar joy",
    "jamoat",
    "ombor",
    "sanoat",
    "maktab",
    "bogcha",
    "kasalxona",
    "ofis",
    "binosi",
    "obyekt",
    "obekt",
]

REGION_KEYWORDS = [
    "shahar",
    "qishloq",
    "sanoat zona",
    "hudud",
    "seysmik",
    "iqlim",
]

PHASE_KEYWORDS = [
    "loyihalash",
    "qurish",
    "qurilish",
    "ekspluatatsiya",
    "montaj",
    "rekonstruksiya",
]

TECH_PARAM_KEYWORDS = [
    "masofa",
    "balandlik",
    "foiz",
    "vaqt",
    "muddat",
    "maydon",
    "eni",
    "uzunligi",
    "kenglik",
    "qalinlik",
    "diametr",
    "hajm",
]

COMPARISON_KEYWORDS = ["qaysi biri", "taqqos", "solishtir", "farqi", "to'g'riroq", "togriroq"]
EXCEPTION_KEYWORDS = ["istisno", "maxsus holat", "alohida holat", "cheklov"]
REFERENCE_NEED_HINTS = ["izoh", "tushuntir", "qisqacha"]
REFERENCE_SPECIFIED_HINTS = ["band", "modda", "rasmiy", "havola", "manba"]
EXAMPLE_HINTS = ["misol", "amaliy"]
PURPOSE_HINTS = ["nazorat", "loyiha", "ekspertiza", "tekshiruv", "kelishish", "tasdiq"]
TABLE_HINTS = ["jadval", "table"]
DOC_CODE_RE = re.compile(r"\b(shnq|qmq|kmk|snip)\s*([0-9][0-9.\-]*)\b", re.IGNORECASE)
TABLE_TYPO_RE = re.compile(r"\b(jadval|jadvl|jadvlda|jdval|table)\b", re.IGNORECASE)
CONTEXT_SENSITIVE_TERMS = ["eshik", "deraza", "zina", "yo'lak", "yolak", "evakuatsiya", "kenglik", "balandlik", "masofa"]
USE_CASE_KEYWORDS = [
    "turar joy",
    "aholi yashash",
    "jamoat",
    "yongin",
    "yong'in",
    "sanoat",
    "nogiron",
    "evakuatsiya",
    "ombor",
]
TABLE_NUMBER_RE = re.compile(
    r"(?:\bjadval(?:da|ni|ga|dan|ning|lar)?\s*[-.]?\s*(\d+[a-z]?)\b|"
    r"\b(\d+[a-z]?)\s*[-.]?\s*jadval(?:da|ni|ga|dan|ning|lar)?\b)",
    re.IGNORECASE,
)
TABLE_NUM_ONLY_RE = re.compile(r"\b(\d+[a-z]?)\s*[-.]?\s*(?:jadval|jadvl|jadvlda|jdval)\b", re.IGNORECASE)
TABLE_CONTEXT_STOP_WORDS = {
    "jadval",
    "jadvlda",
    "jadvl",
    "jdval",
    "table",
    "haqida",
    "malumot",
    "ma'lumot",
    "nima",
    "deyilgan",
    "ber",
    "bering",
    "korsat",
    "ko'rsat",
    "qaysi",
    "boyicha",
    "bo'yicha",
    "shnq",
    "qmq",
    "kmk",
    "snip",
}
TABLE_QUESTION_HINTS = {
    "nima",
    "qanday",
    "qancha",
    "necha",
    "qaysi",
    "nimaga",
    "nega",
    "izoh",
    "izohla",
    "tushuntir",
    "tushuntirib",
    "hisobla",
    "ber",
    "degan",
    "kerak",
}

CLARIFICATION_RULES = [
    ("missing_document", "Qaysi hujjat nazarda tutilmoqda? (masalan: SHNQ 2.08.01-24)"),
    ("missing_clause", "Aniq qaysi band yoki modda haqida so'rayapsiz?"),
    ("missing_object_type", "Qaysi obyekt turi uchun?"),
    ("missing_region", "Qaysi hudud sharoitida?"),
    ("missing_phase", "Qurilishning qaysi bosqichi nazarda tutilgan?"),
    ("missing_edition", "Hujjatning qaysi yildagi tahriri kerak?"),
    ("missing_parameter", "Aniq qaysi parametr haqida?"),
    ("missing_comparison_target", "Qaysi ikki talabni solishtirmoqchisiz?"),
    ("missing_exception_state", "Bu maxsus yoki istisno holatmi?"),
    ("missing_reference_mode", "Faqat izohmi yoki rasmiy band bilanmi?"),
    ("missing_example_mode", "Amaliy misol bilan tushuntiraymi?"),
    ("missing_purpose", "Bu ma'lumot qaysi maqsad uchun kerak?"),
    (
        "missing_use_case",
        "Aynan qaysi holat uchun? (masalan: turar joy, jamoat binosi yoki yong'in xavfsizligi)",
    ),
]


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
    has_table_word = bool(TABLE_TYPO_RE.search(normalized))
    return (has_doc and has_table_num) or (has_table_num and has_table_word) or _contains_any(normalized, TABLE_HINTS)


def _extract_table_number(text: str):
    normalized = _normalize_text(text)
    match = TABLE_NUMBER_RE.search(normalized)
    if not match:
        match = TABLE_NUM_ONLY_RE.search(normalized)
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
    return (
        f"{table.document.code} bo'yicha {table.table_number}-jadval topildi "
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


def _build_table_qa_answer(message: str, table: NormTable) -> str:
    system = (
        "Siz SHNQ jadvali bo'yicha yordamchisiz. Faqat berilgan jadval matniga tayangan holda javob bering. "
        "Jadvalda aniq topilmasa, shu holatni ochiq ayting va taxmin qilmang."
    )
    prompt = (
        f"Savol: {message}\n\n"
        f"Hujjat: {table.document.code}\n"
        f"Bo'lim: {table.section_title or (table.chapter.title if table.chapter else '-')}\n"
        f"Jadval raqami: {table.table_number}\n"
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
    # Jadvalni faqat jadvalga oid aniq so'rovda qaytaramiz.
    if not _is_table_request(message):
        return None
    table_number = _extract_table_number(message)
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
        qs = [t for t in qs if (t.table_number or "").lower() == table_number.lower()]

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

    total = Clause.objects.count()
    if total == 0:
        return
    existing = ClauseEmbedding.objects.filter(embedding_model=EMBEDDING_MODEL).count()

    needs_upsert = existing < total
    if USE_QDRANT:
        qdrant_total = qdrant_count_points()
        if qdrant_total < existing:
            needs_upsert = True

    if not needs_upsert:
        return

    upsert_clause_embeddings(embedding_model=EMBEDDING_MODEL, force_update=False)
    _EMBED_CACHE_MODEL = None
    _EMBED_CACHE_DATA = None


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


def _score_with_qdrant(query_vec, query_terms, requested_doc_code=None):
    if not query_vec:
        return []
    limit = RAG_QDRANT_DOC_LIMIT if requested_doc_code else RAG_QDRANT_LIMIT
    hits = qdrant_search(query_vec, limit=limit, doc_code=requested_doc_code)
    if not hits:
        return []

    ids = [str(hit.id) for hit in hits]
    embeddings = ClauseEmbedding.objects.select_related("clause", "clause__document", "clause__chapter").filter(
        clause_id__in=ids
    )
    emb_map = {str(emb.clause_id): emb for emb in embeddings}

    scored = []
    for hit in hits:
        emb = emb_map.get(str(hit.id))
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
    # Harflar/raqamlar/apostrof bilan so'zlarni ajratamiz.
    terms = re.findall(r"[a-z0-9']+", normalized)
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
    hits = 0
    for term in terms:
        if term in clause_text:
            hits += 1
            continue
        if term in chapter or term in shnq_code:
            hits += 1
    return hits / max(len(terms), 1)


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


def _build_rag_prompt(question, sources, response_language="uz"):
    context_chunks = []
    for idx, emb in enumerate(sources, 1):
        clause = emb.clause
        header = f"Manba {idx}"
        lines = [
            header,
            f"Hujjat: {emb.shnq_code}",
            f"Bob: {emb.chapter_title or 'Nomalum bob'}",
            f"Band: {emb.clause_number or '-'}",
            f"Matn: {clause.text}",
        ]
        context_chunks.append("\n".join(lines))

    context = "\n\n".join(context_chunks)
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
    prompt = f"Savol: {question}\n\nKontekst:\n{context}\n\nJavob:"
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
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if "Batafsil:" not in text:
        text = f"Batafsil: {text}"
    if "Qisqa qilib aytganda:" not in text:
        # Qisqa xulosani model bermagan bo'lsa, mavjud matndan ixcham yakun qo'shamiz.
        short = text.replace("Batafsil:", "").strip()
        short = short.split(".")[0].strip()
        if short:
            text = f"{text}\nQisqa qilib aytganda: {short}."
    return text


class ChatAPIView(APIView):
    authentication_classes = []
    permission_classes = []

    def finalize_response(self, request, response, *args, **kwargs):
        response = super().finalize_response(request, response, *args, **kwargs)
        language = getattr(request, "_query_language", "uz")
        timings = getattr(request, "_timings", None)
        request_start = getattr(request, "_request_start", None)
        if isinstance(getattr(response, "data", None), dict) and "answer" in response.data:
            try:
                t0 = time.perf_counter()
                response.data["answer"] = ensure_answer_language(response.data["answer"], language)
                if isinstance(timings, dict):
                    timings["translate_out"] = round((time.perf_counter() - t0) * 1000, 2)
            except Exception:
                pass
            meta = response.data.get("meta")
            if isinstance(meta, dict):
                meta.setdefault("query_language", language)
                if isinstance(timings, dict):
                    stage_order = ["detect", "translate_in", "embed", "rag_generate", "translate_out"]
                    ms = {k: round(float(timings.get(k, 0.0)), 2) for k in stage_order}
                    active = {k: v for k, v in ms.items() if v > 0}
                    if active:
                        slowest_stage = max(active, key=active.get)
                        meta["timings_ms"] = ms
                        meta["slowest_stage"] = slowest_stage
                        meta["slowest_stage_ms"] = active[slowest_stage]
                    if request_start is not None:
                        total_ms = round((time.perf_counter() - request_start) * 1000, 2)
                        meta["total_ms"] = total_ms
                        logger.info(
                            "chat_timing total_ms=%s slowest=%s slowest_ms=%s timings=%s",
                            total_ms,
                            meta.get("slowest_stage"),
                            meta.get("slowest_stage_ms"),
                            ms,
                        )
            table_html = response.data.get("table_html")
            if isinstance(table_html, str) and table_html.strip() and language in {"en", "ru", "ko"}:
                try:
                    response.data["table_html"] = translate_html_preserving_tags(
                        table_html,
                        target_language=language,
                        source_language="uz",
                    )
                except Exception:
                    pass
            sources = response.data.get("sources")
            if isinstance(sources, list) and language in {"en", "ru", "ko"}:
                for item in sources:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "table":
                        continue
                    html_value = item.get("html")
                    if isinstance(html_value, str) and html_value.strip():
                        try:
                            item["html"] = translate_html_preserving_tags(
                                html_value,
                                target_language=language,
                                source_language="uz",
                            )
                        except Exception:
                            pass
                    md_value = item.get("markdown")
                    if isinstance(md_value, str) and md_value.strip():
                        try:
                            item["markdown"] = ensure_answer_language(md_value, language)
                        except Exception:
                            pass
        return response

    def post(self, request):
        total_start = time.perf_counter()
        timings = {
            "detect": 0.0,
            "translate_in": 0.0,
            "embed": 0.0,
            "rag_generate": 0.0,
            "translate_out": 0.0,
        }
        request._timings = timings
        request._request_start = total_start

        message = (request.data.get("message") or "").strip()
        if not message:
            return Response({"error": "message is required"}, status=status.HTTP_400_BAD_REQUEST)

        original_message = message
        search_message = message
        message_language = "uz"
        try:
            t_detect = time.perf_counter()
            message_language = detect_query_language(message)
            timings["detect"] = round((time.perf_counter() - t_detect) * 1000, 2)
            if message_language in {"en", "ru", "ko"}:
                t_translate = time.perf_counter()
                search_message = translate_query_for_search(message, message_language)
                timings["translate_in"] = round((time.perf_counter() - t_translate) * 1000, 2)
            else:
                search_message = message
        except Exception:
            search_message = message
            message_language = "uz"
        request._query_language = message_language

        # Guardrails: salomlashuv va mavzudan tashqari savollarni LLM/RAGdan oldin ushlaymiz.
        if _is_greeting(search_message):
            return Response(
                {
                    "answer": _build_greeting_response(),
                    "sources": [],
                    "meta": {"type": "greeting", "model": CHAT_MODEL},
                }
            )

        # Aniq mavzudan tashqari savollarni RAGga yubormaymiz.
        if _is_clearly_out_of_scope(search_message) and not _is_shnq_related(search_message):
            return Response(
                {
                    "answer": _build_out_of_scope_response(),
                    "sources": [],
                    "meta": {"type": "out_of_scope", "model": CHAT_MODEL},
                }
            )

        if _is_table_request(search_message):
            table, table_number, doc_code, candidates = _find_table_for_query(search_message)
            if not table_number:
                return Response(
                    {
                        "answer": "Qaysi jadval nazarda tutilmoqda? (masalan: 9-jadval)",
                        "sources": [],
                        "meta": {"type": "clarification", "missing_case": "missing_table_number", "model": CHAT_MODEL},
                    }
                )

            if not doc_code:
                docs = _table_candidate_docs(table_number)
                if len(docs) == 1 and table:
                    doc_code = docs[0]
                else:
                    hint = f" Mavjudlari: {', '.join(docs)}." if docs else ""
                    return Response(
                        {
                            "answer": f"{table_number}-jadval qaysi hujjatda kerak? (masalan: SHNQ 2.07.01-23).{hint}",
                            "sources": [],
                            "meta": {
                                "type": "clarification",
                                "missing_case": "missing_document_for_table",
                                "model": CHAT_MODEL,
                                "candidate_documents": docs,
                            },
                        }
                    )

            if not table and candidates:
                chapters = _table_candidate_chapters(candidates)
                chapter_hint = f" Variantlar: {', '.join(chapters)}." if chapters else ""
                return Response(
                    {
                        "answer": f"{table_number}-jadval qaysi bo'lim/bob bo'yicha kerak?{chapter_hint}",
                        "sources": [],
                        "meta": {
                            "type": "clarification",
                            "missing_case": "missing_table_chapter_context",
                            "model": CHAT_MODEL,
                            "candidate_chapters": chapters,
                        },
                    }
                )

            if not table:
                doc_label = doc_code or "ko'rsatilgan hujjat"
                return Response(
                    {
                        "answer": f"{doc_label} bo'yicha {table_number}-jadval topilmadi.",
                        "sources": [],
                        "meta": {"type": "no_match", "model": CHAT_MODEL, "target": "table"},
                    }
                )

            if _is_table_direct_lookup_request(search_message):
                answer = _build_table_answer(table)
            else:
                t_rag = time.perf_counter()
                answer = _build_table_qa_answer(search_message, table)
                timings["rag_generate"] = round((time.perf_counter() - t_rag) * 1000, 2)
            sources = [
                {
                    "type": "table",
                    "shnq_code": table.document.code,
                    "chapter": table.chapter.title if table.chapter else None,
                    "table_number": table.table_number,
                    "title": table.title,
                    "html_anchor": table.html_anchor,
                    "markdown": table.markdown,
                    "html": table.raw_html,
                }
            ]
            QuestionAnswer.objects.create(
                question=original_message,
                answer=answer,
                top_clause_ids=[],
            )
            return Response(
                {
                    "answer": answer,
                    "sources": sources,
                    "table_html": table.raw_html,
                    "meta": {"type": "table_lookup", "model": CHAT_MODEL},
                }
            )

        try:
            _ensure_embeddings()
        except Exception as exc:
            return Response({"error": f"Embedding tayyorlash xatoligi: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        rewritten_message = _rewrite_query_if_needed(search_message)

        try:
            t_embed = time.perf_counter()
            query_vec = embed_text(rewritten_message, model=EMBEDDING_MODEL)
            timings["embed"] = round((time.perf_counter() - t_embed) * 1000, 2)
        except Exception as exc:
            return Response({"error": f"Embedding xatoligi: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        requested_doc_code = _extract_doc_code(search_message)
        query_terms = _extract_query_terms(rewritten_message)

        if USE_QDRANT:
            try:
                scored = _score_with_qdrant(query_vec, query_terms, requested_doc_code=requested_doc_code)
            except Exception as exc:
                return Response({"error": f"Qdrant xatoligi: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            if requested_doc_code and not scored:
                return Response(
                    {
                        "answer": f"{requested_doc_code} bo'yicha mos band topilmadi.",
                        "sources": [],
                        "meta": {
                            "type": "no_match",
                            "target": "document",
                            "model": CHAT_MODEL,
                            "requested_document": requested_doc_code,
                        },
                    }
                )
        else:
            embeddings = _get_embeddings_for_query()
            if requested_doc_code:
                embeddings = _filter_embeddings_by_doc_code(embeddings, requested_doc_code)
                if not embeddings:
                    return Response(
                        {
                            "answer": f"{requested_doc_code} bo'yicha mos band topilmadi.",
                            "sources": [],
                            "meta": {
                                "type": "no_match",
                                "target": "document",
                                "model": CHAT_MODEL,
                                "requested_document": requested_doc_code,
                            },
                        }
                    )

            scored = []
            for emb in embeddings:
                semantic = cosine_similarity(query_vec, emb.vector)
                keyword = _keyword_score(query_terms, emb)
                score = semantic + (KEYWORD_WEIGHT * keyword)
                scored.append((score, emb, semantic, keyword))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score = scored[0][0] if scored else 0.0
        if not requested_doc_code:
            ask_doc_clarification, doc_candidates = _should_ask_document_clarification(scored, best_score)
            if ask_doc_clarification:
                return Response(
                    {
                        "answer": _build_document_clarification_answer(doc_candidates),
                        "sources": [],
                        "meta": {
                            "type": "clarification",
                            "missing_case": "ambiguous_document",
                            "model": CHAT_MODEL,
                            "best_score": round(best_score, 4),
                            "candidate_documents": doc_candidates,
                        },
                    }
                )

        allow_relaxed = _can_answer_with_relaxed_threshold(scored, best_score)
        if best_score < STRICT_MIN_SCORE:
            if allow_relaxed:
                pass
            else:
                clarification = _needs_clarification(search_message)
                if clarification:
                    code, question = clarification
                    return Response(
                        {
                            "answer": question,
                            "sources": [],
                            "meta": {
                                "type": "clarification",
                                "missing_case": code,
                                "model": CHAT_MODEL,
                                "best_score": round(best_score, 4),
                                "strict_min_score": STRICT_MIN_SCORE,
                            },
                        }
                    )
                return Response(
                    {
                        "answer": "Mos band topilmadi.",
                        "sources": [],
                        "meta": {
                            "type": "no_match",
                            "model": CHAT_MODEL,
                            "best_score": round(best_score, 4),
                            "strict_min_score": STRICT_MIN_SCORE,
                            "rewritten": rewritten_message != search_message,
                        },
                    }
                )

        filtered = [
            (score, item, semantic, keyword)
            for score, item, semantic, keyword in scored
            if score >= MIN_SCORE
        ]
        candidate_pairs = filtered[: max(RERANK_CANDIDATES, 5)]
        reranked = _llm_rerank(search_message, candidate_pairs)
        top_pairs = reranked[:5]
        top = [item for _, item, *_rest in top_pairs]

        if not top:
            clarification = _needs_clarification(search_message)
            if clarification:
                code, question = clarification
                return Response(
                    {
                        "answer": question,
                        "sources": [],
                        "meta": {"type": "clarification", "missing_case": code, "model": CHAT_MODEL},
                    }
                )
            return Response(
                {
                    "answer": "Mos band topilmadi.",
                    "sources": [],
                    "meta": {"type": "no_match", "model": CHAT_MODEL},
                }
            )

        system, prompt = _build_rag_prompt(search_message, top, response_language=message_language)
        try:
            t_rag = time.perf_counter()
            answer = generate_text(
                prompt,
                system=system,
                model=CHAT_MODEL,
                options={"temperature": 0.0, "top_p": 0.9, "max_tokens": RAG_FINAL_MAX_TOKENS},
            )
            timings["rag_generate"] = round((time.perf_counter() - t_rag) * 1000, 2)
        except Exception as exc:
            answer = f"LLM javobida xatolik: {exc}"
        if not answer:
            answer = top[0].clause.text
        answer = _cleanup_answer_format(answer)
        sources = []
        for score, emb, semantic, keyword in top_pairs:
            clause = emb.clause
            sources.append(
                {
                    "shnq_code": emb.shnq_code,
                    "chapter": emb.chapter_title,
                    "clause_number": emb.clause_number,
                    "html_anchor": clause.html_anchor,
                    "lex_url": emb.lex_url,
                    "snippet": clause.text[:280],
                    "score": round(score, 4),
                    "semantic_score": round(semantic, 4),
                    "keyword_score": round(keyword, 4),
                }
            )

        related_table = _pick_related_table_from_rag(search_message, top_pairs)
        table_html = None
        if related_table:
            table_html = related_table.raw_html
            sources.append(
                {
                    "type": "table",
                    "shnq_code": related_table.document.code,
                    "chapter": related_table.section_title
                    or (related_table.chapter.title if related_table.chapter else None),
                    "table_number": related_table.table_number,
                    "title": related_table.title,
                    "html_anchor": related_table.html_anchor,
                    "markdown": related_table.markdown,
                    "html": related_table.raw_html,
                }
            )

        QuestionAnswer.objects.create(
            question=original_message,
            answer=answer,
            top_clause_ids=[str(emb.clause_id) for emb in top],
        )

        return Response(
            {
                "answer": answer,
                "sources": sources,
                "table_html": table_html,
                "meta": {
                    "type": "rag",
                    "model": CHAT_MODEL,
                    "answer_language": message_language,
                    "min_score": MIN_SCORE,
                    "strict_min_score": STRICT_MIN_SCORE,
                    "relaxed_threshold_used": best_score < STRICT_MIN_SCORE and allow_relaxed,
                    "keyword_weight": KEYWORD_WEIGHT,
                    "rerank_enabled": RERANK_ENABLED,
                    "rerank_candidates": RERANK_CANDIDATES,
                    "rewritten": rewritten_message != search_message,
                    "query_used": rewritten_message,
                    "query_original": original_message,
                },
            }
        )


