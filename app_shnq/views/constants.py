import logging
import os
import re
import time
import unicodedata

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from ..deepseek_client import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    detect_query_language,
    embed_text,
    ensure_answer_language,
    generate_text,
    translate_query_for_search,
    translate_html_preserving_tags,
)
from ..embeddings import cosine_similarity, upsert_clause_embeddings
from ..models import Clause, ClauseEmbedding, NormTable, QuestionAnswer
from ..qdrant_store import count_points as qdrant_count_points, search as qdrant_search


EMBEDDING_MODEL = os.getenv("OPENAI_EMBED_MODEL", os.getenv("DEEPSEEK_EMBED_MODEL", DEFAULT_EMBED_MODEL))
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", os.getenv("DEEPSEEK_CHAT_MODEL", DEFAULT_CHAT_MODEL))
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
RAG_MULTILINGUAL_NATIVE_FIRST = os.getenv("RAG_MULTILINGUAL_NATIVE_FIRST", "1") == "1"
RAG_MULTILINGUAL_TRANSLATE_FALLBACK = os.getenv("RAG_MULTILINGUAL_TRANSLATE_FALLBACK", "1") == "1"
RAG_TRANSLATION_FALLBACK_THRESHOLD = float(
    os.getenv("RAG_TRANSLATION_FALLBACK_THRESHOLD", str(STRICT_MIN_SCORE))
)
RAG_TRANSLATED_QUERY_SCORE_WEIGHT = float(os.getenv("RAG_TRANSLATED_QUERY_SCORE_WEIGHT", "0.97"))

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
TABLE_HINTS = ["jadval", "table", "ilova", "appendix"]
DOC_CODE_RE = re.compile(r"\b(shnq|qmq|kmk|snip)\s*([0-9][0-9.\-]*)\b", re.IGNORECASE)
TABLE_TYPO_RE = re.compile(r"\b(jadval|jadvl|jadvlda|jdval|table|ilova|appendix)\b", re.IGNORECASE)
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
APPENDIX_NUMBER_RE = re.compile(
    r"(?:\b(\d+)\s*[-.]?\s*ilova(?:si|da|ga|dan|ning|lar)?\b|"
    r"\bilova(?:si|da|ga|dan|ning|lar)?\s*[-.]?\s*(\d+)\b)",
    re.IGNORECASE,
)
TABLE_CONTEXT_STOP_WORDS = {
    "jadval",
    "jadvlda",
    "jadvl",
    "jdval",
    "table",
    "ilova",
    "appendix",
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


