import hashlib
import json
import math
import os
import re
import threading
from typing import Any

from openai import OpenAI


DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_EMBED_MODEL = os.getenv("DEEPSEEK_EMBED_MODEL", "deepseek-embedding")
DEFAULT_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
DEFAULT_EMBED_DIM = int(os.getenv("DEEPSEEK_EMBED_DIM", "768"))
DEFAULT_CHAT_TIMEOUT = int(os.getenv("DEEPSEEK_CHAT_TIMEOUT", "45"))
DEFAULT_EMBED_TIMEOUT = int(os.getenv("DEEPSEEK_EMBED_TIMEOUT", "12"))
DEFAULT_TRANSLATE_TIMEOUT = int(os.getenv("DEEPSEEK_TRANSLATE_TIMEOUT", "30"))
DEFAULT_TRANSLATE_MAX_TOKENS = int(os.getenv("DEEPSEEK_TRANSLATE_MAX_TOKENS", "320"))
DEFAULT_BATCH_TRANSLATE_MAX_TOKENS = int(os.getenv("DEEPSEEK_BATCH_TRANSLATE_MAX_TOKENS", "900"))
FAST_LANGUAGE_DETECT = os.getenv("DEEPSEEK_FAST_LANGUAGE_DETECT", "1") == "1"
EMBEDDING_FALLBACK = os.getenv("DEEPSEEK_EMBED_FALLBACK", "semantic_hash")
STRICT_EMBEDDING = os.getenv("DEEPSEEK_EMBED_STRICT", "0") == "1"
TRANSLATE_PROVIDER = os.getenv("TRANSLATE_PROVIDER", "deepseek").strip().lower()
NLLB_MODEL_NAME = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-1.3B").strip()
NLLB_DEVICE = os.getenv("NLLB_DEVICE", "cpu").strip().lower()
NLLB_BATCH_SIZE = int(os.getenv("NLLB_BATCH_SIZE", "8"))
NLLB_MAX_NEW_TOKENS = int(os.getenv("NLLB_MAX_NEW_TOKENS", "256"))
NLLB_NUM_THREADS = int(os.getenv("NLLB_NUM_THREADS", "0"))
NLLB_NUM_INTEROP_THREADS = int(os.getenv("NLLB_NUM_INTEROP_THREADS", "0"))

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_EMBED_MODEL_SUPPORT = {}
_SUPPORTED_LANGS = {"uz", "en", "ru", "ko"}
_LANG_LABELS = {
    "uz": "Uzbek",
    "en": "English",
    "ru": "Russian",
    "ko": "Korean",
    "auto": "auto",
}
_EN_HINT_WORDS = {
    "what",
    "which",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "the",
    "and",
    "or",
    "for",
    "with",
    "shall",
    "must",
    "requirements",
    "fire",
    "safety",
}
_UZ_HINT_WORDS = {
    "va",
    "uchun",
    "qanday",
    "qaysi",
    "nima",
    "necha",
    "kerak",
    "bo'yicha",
    "bilan",
    "yoki",
    "hujjat",
    "band",
    "jadval",
    "masofa",
    "balandlik",
    "kenglik",
    "qurilish",
    "me'yor",
    "talab",
}
_NLLB_LANGS = {
    "uz": "uzn_Latn",
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "ko": "kor_Hang",
}
_NLLB_LOCK = threading.Lock()
_NLLB_STATE = {
    "tokenizer": None,
    "model": None,
    "torch": None,
    "device": None,
    "error": None,
}


def _script_counts(text: str):
    value = text or ""
    hangul = sum(1 for ch in value if 0xAC00 <= ord(ch) <= 0xD7AF)
    cyrillic = sum(1 for ch in value if 0x0400 <= ord(ch) <= 0x04FF)
    ascii_count = sum(1 for ch in value if ord(ch) < 128)
    return hangul, cyrillic, ascii_count, len(value)


def _is_strong_english_text(text: str) -> bool:
    hangul, cyrillic, ascii_count, total = _script_counts(text)
    if total == 0 or hangul > 0 or cyrillic > 0:
        return False
    tokens = re.findall(r"[a-zA-Z']+", (text or "").lower())
    if len(tokens) < 3:
        return False
    en_hits = sum(1 for t in tokens if t in _EN_HINT_WORDS)
    ascii_ratio = ascii_count / max(total, 1)
    return en_hits >= 1 and ascii_ratio > 0.95


def _parse_language_token(raw: str) -> str | None:
    value = (raw or "").strip().lower()
    if not value:
        return None
    compact = re.sub(r"[^a-z]", "", value)
    if compact.startswith("en") or "english" in compact:
        return "en"
    if compact.startswith("uz") or "uzbek" in compact:
        return "uz"
    if compact.startswith("ru") or "russian" in compact:
        return "ru"
    if compact.startswith("ko") or "korean" in compact:
        return "ko"
    return None


def _resolve_api_key(api_key=None):
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY topilmadi. Muhit ozgaruvchisiga API kalitni kiriting.")
    return key


def _client(api_key=None, base_url=None, timeout=DEFAULT_CHAT_TIMEOUT):
    return OpenAI(
        api_key=_resolve_api_key(api_key),
        base_url=(base_url or DEFAULT_BASE_URL).rstrip("/"),
        timeout=timeout,
    )


def _extract_message_content(message: Any) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if not content:
        # DeepSeek reasoner modelida ayrim javoblar reasoning_content maydonida keladi.
        reasoning = getattr(message, "reasoning_content", "")
        return (reasoning or "").strip()
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                text = chunk.get("text")
            else:
                text = getattr(chunk, "text", None)
            if text:
                parts.append(str(text))
        return "".join(parts).strip()
    return str(content).strip()


def _hash_embedding(text: str, dim: int = DEFAULT_EMBED_DIM):
    vec = [0.0] * dim
    for token in _TOKEN_RE.findall((text or "").lower()):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % dim
        vec[idx] += 1.0

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def generate_text(
    prompt,
    system=None,
    model=None,
    options=None,
    base_url=None,
    timeout=DEFAULT_CHAT_TIMEOUT,
    api_key=None,
):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    req = {
        "model": model or DEFAULT_CHAT_MODEL,
        "messages": messages,
        "stream": False,
    }
    if options:
        for key in ("temperature", "top_p", "max_tokens"):
            if key in options:
                req[key] = options[key]

    response = _client(api_key=api_key, base_url=base_url, timeout=timeout).chat.completions.create(**req)
    if not response.choices:
        return ""
    return _extract_message_content(response.choices[0].message)


def _semantic_signature(text, base_url=None, timeout=DEFAULT_CHAT_TIMEOUT, api_key=None):
    system = (
        "Siz semantik indekslash yordamchisisiz. Berilgan matndan faqat mazmunni ifodalovchi "
        "asosiy kalit iboralarni ajrating va ularni bitta satrda, vergul bilan qaytaring. "
        "Yangi fakt qoshmang."
    )
    prompt = f"Matn:\n{text}\n\nSemantik kalit iboralar:"
    return generate_text(
        prompt,
        system=system,
        model=DEFAULT_CHAT_MODEL,
        options={"temperature": 0.0, "top_p": 0.9},
        base_url=base_url,
        timeout=timeout,
        api_key=api_key,
    )


def embed_text(text, model=None, base_url=None, timeout=DEFAULT_EMBED_TIMEOUT, api_key=None):
    if not text:
        return []

    model_name = model or DEFAULT_EMBED_MODEL
    client = _client(api_key=api_key, base_url=base_url, timeout=timeout)

    if _EMBED_MODEL_SUPPORT.get(model_name, True):
        try:
            response = client.embeddings.create(model=model_name, input=text)
            if response.data and response.data[0].embedding:
                _EMBED_MODEL_SUPPORT[model_name] = True
                return response.data[0].embedding
        except Exception:
            _EMBED_MODEL_SUPPORT[model_name] = False
            if STRICT_EMBEDDING:
                raise

    seed_text = text
    if EMBEDDING_FALLBACK == "semantic_hash":
        try:
            signature = _semantic_signature(text, base_url=base_url, timeout=timeout, api_key=api_key)
            if signature:
                seed_text = signature
        except Exception:
            pass
    return _hash_embedding(seed_text)


def _heuristic_detect_language(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized:
        return "uz"

    hangul_count, cyrillic_count, ascii_count, total = _script_counts(normalized)
    if hangul_count > 0 and hangul_count >= cyrillic_count:
        return "ko"
    if cyrillic_count > 0:
        return "ru"

    tokens = re.findall(r"[a-zA-Z']+", normalized.lower())
    if not tokens:
        return "uz"

    uz_hits = sum(1 for t in tokens if t in _UZ_HINT_WORDS)
    # Uzbek lotinidagi maxsus belgilar va ko'p uchraydigan funksional so'zlar.
    if any(ch in normalized for ch in ("o'", "g'", "o‘", "g‘", "ʻ")) or uz_hits >= 2:
        return "uz"

    en_hits = sum(1 for t in tokens if t in _EN_HINT_WORDS)
    ascii_ratio = ascii_count / max(total, 1)
    if en_hits >= 2 or (ascii_ratio > 0.98 and len(tokens) >= 4):
        return "en"
    return "uz"


def detect_query_language(
    text: str,
    model=None,
    base_url=None,
    timeout=DEFAULT_TRANSLATE_TIMEOUT,
    api_key=None,
) -> str:
    if not (text or "").strip():
        return "uz"
    heuristic = _heuristic_detect_language(text)
    if FAST_LANGUAGE_DETECT:
        return heuristic
    if heuristic in {"ko", "ru"}:
        return heuristic
    if heuristic == "en":
        # Kuchli inglizcha signal bo'lsa, noto'g'ri "ru" klassifikatsiyasini oldini olamiz.
        return "en"

    system = (
        "You are a strict language classifier. "
        "Return only one token from this set: "
        "'uz' for Uzbek, 'en' for English, 'ru' for Russian, 'ko' for Korean."
    )
    prompt = (
        "Detect the language of this user query.\n"
        "If mixed, choose the dominant language.\n"
        f"Query: {text}\n\nLanguage:"
    )
    try:
        raw = generate_text(
            prompt,
            system=system,
            model=model or DEFAULT_CHAT_MODEL,
            options={"temperature": 0.0, "top_p": 0.9, "max_tokens": 4},
            base_url=base_url,
            timeout=timeout,
            api_key=api_key,
        )
        parsed = _parse_language_token(raw)
        if parsed in _SUPPORTED_LANGS:
            # Heuristika bilan ziddiyatda bo'lsa, script-ga asoslangan natijani ustun qo'yamiz.
            if heuristic in {"ko", "ru"} and parsed != heuristic:
                return heuristic
            if heuristic == "en" and parsed in {"ru", "ko"}:
                return "en"
            return parsed
    except Exception:
        pass
    return heuristic


def _resolve_nllb_source_lang(text: str, source_language: str) -> str:
    if source_language and source_language != "auto":
        return source_language
    return detect_query_language(text)


def _load_nllb():
    if _NLLB_STATE["tokenizer"] is not None or _NLLB_STATE["error"] is not None:
        return
    with _NLLB_LOCK:
        if _NLLB_STATE["tokenizer"] is not None or _NLLB_STATE["error"] is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            if NLLB_NUM_THREADS > 0:
                torch.set_num_threads(NLLB_NUM_THREADS)
            if NLLB_NUM_INTEROP_THREADS > 0:
                torch.set_num_interop_threads(NLLB_NUM_INTEROP_THREADS)

            device = NLLB_DEVICE
            if device != "cpu" and device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"

            tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
            model.eval()
            if device != "cpu":
                model.to(device)

            _NLLB_STATE["tokenizer"] = tokenizer
            _NLLB_STATE["model"] = model
            _NLLB_STATE["torch"] = torch
            _NLLB_STATE["device"] = device
        except Exception as exc:
            _NLLB_STATE["error"] = exc


def _nllb_translate_batch(texts, target_language: str, source_language: str = "auto"):
    if not texts:
        return []

    src = _resolve_nllb_source_lang(texts[0], source_language)
    if src not in _NLLB_LANGS or target_language not in _NLLB_LANGS:
        raise ValueError("NLLB language not supported")

    _load_nllb()
    if _NLLB_STATE["error"] is not None:
        raise RuntimeError(str(_NLLB_STATE["error"]))

    tokenizer = _NLLB_STATE["tokenizer"]
    model = _NLLB_STATE["model"]
    torch = _NLLB_STATE["torch"]
    device = _NLLB_STATE["device"]

    src_code = _NLLB_LANGS[src]
    tgt_code = _NLLB_LANGS[target_language]
    tokenizer.src_lang = src_code

    results = []
    batch_size = max(1, int(NLLB_BATCH_SIZE))
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        if device and device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
                max_new_tokens=NLLB_MAX_NEW_TOKENS,
                num_beams=1,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        results.extend([item.strip() for item in decoded])
    return results


def _nllb_translate_text(text: str, target_language: str, source_language: str = "auto") -> str:
    outputs = _nllb_translate_batch([text], target_language=target_language, source_language=source_language)
    return outputs[0] if outputs else text


def translate_text(
    text: str,
    target_language: str,
    source_language: str = "auto",
    model=None,
    base_url=None,
    timeout=DEFAULT_TRANSLATE_TIMEOUT,
    api_key=None,
) -> str:
    payload = (text or "").strip()
    if not payload:
        return payload
    if source_language and target_language and source_language != "auto":
        if source_language == target_language:
            return payload

    if TRANSLATE_PROVIDER == "nllb":
        try:
            translated = _nllb_translate_text(payload, target_language=target_language, source_language=source_language)
            if translated and translated.strip():
                return translated.strip()
        except Exception:
            pass

    source_label = _LANG_LABELS.get(source_language, "auto")
    target_label = _LANG_LABELS.get(target_language, "Uzbek")
    system = (
        "You are a professional translator. "
        "Translate accurately and naturally. "
        "Preserve SHNQ/QMQ/SNIP codes, numbers, and units exactly. "
        "Return only the translated text without explanations."
    )
    prompt = (
        f"Source language: {source_label}\n"
        f"Target language: {target_label}\n"
        f"Text:\n{payload}\n\nTranslated text:"
    )
    try:
        translated = generate_text(
            prompt,
            system=system,
            model=model or DEFAULT_CHAT_MODEL,
            options={"temperature": 0.0, "top_p": 0.9, "max_tokens": DEFAULT_TRANSLATE_MAX_TOKENS},
            base_url=base_url,
            timeout=timeout,
            api_key=api_key,
        )
        if translated and translated.strip():
            return translated.strip()
    except Exception:
        pass
    return payload


def _is_suspicious_search_translation(source_language: str, translated: str) -> bool:
    if not (translated or "").strip():
        return True
    hangul, cyrillic, _ascii_count, _total = _script_counts(translated)
    if source_language == "ko" and hangul > 0:
        return True
    if source_language == "ru" and cyrillic > 0:
        return True
    return False


def _translate_to_uz_for_search(message: str, source_language: str) -> str:
    # EN uchun source aniq, RU/KO uchun auto ancha barqaror ishlaydi.
    primary_source = "en" if source_language == "en" else "auto"
    first = translate_text(message, target_language="uz", source_language=primary_source)
    if not _is_suspicious_search_translation(source_language, first):
        return first

    # Fallback: ko/ru -> en -> uz pivot.
    try:
        pivot_en = translate_text(message, target_language="en", source_language="auto")
        second = translate_text(pivot_en, target_language="uz", source_language="en")
        if not _is_suspicious_search_translation(source_language, second):
            return second
    except Exception:
        pass

    # Oxirgi fallback: LLMdan to'g'ridan-to'g'ri o'zbekcha qidiruv so'rovi yozdiramiz.
    try:
        system = (
            "You rewrite user queries into clear Uzbek (Latin script) for semantic search. "
            "Keep SHNQ/QMQ/SNIP codes, numbers, units, and technical terms exact."
        )
        prompt = (
            f"Original language: {_LANG_LABELS.get(source_language, 'auto')}\n"
            "Rewrite this query into Uzbek (Latin) only:\n"
            f"{message}\n\nUzbek query:"
        )
        forced = generate_text(
            prompt,
            system=system,
            model=DEFAULT_CHAT_MODEL,
            options={"temperature": 0.0, "top_p": 0.9, "max_tokens": 220},
        )
        if forced and forced.strip():
            return forced.strip()
    except Exception:
        pass
    return first


def _strip_code_fences(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _batch_translate_texts(
    texts,
    target_language: str,
    source_language: str = "auto",
    model=None,
    base_url=None,
    timeout=DEFAULT_TRANSLATE_TIMEOUT,
    api_key=None,
):
    if not texts:
        return []

    if TRANSLATE_PROVIDER == "nllb":
        try:
            translated = _nllb_translate_batch(
                list(texts),
                target_language=target_language,
                source_language=source_language,
            )
            if translated and len(translated) == len(texts):
                return [str(item) for item in translated]
        except Exception:
            pass

    source_label = _LANG_LABELS.get(source_language, "auto")
    target_label = _LANG_LABELS.get(target_language, "Uzbek")
    system = (
        "You are a professional translator. "
        "Translate each item in the JSON array. "
        "Return only a valid JSON array of translated strings with the same length and order. "
        "Do not add explanations."
    )
    prompt = (
        f"Source language: {source_label}\n"
        f"Target language: {target_label}\n"
        "Input JSON array:\n"
        f"{json.dumps(texts, ensure_ascii=False)}\n\n"
        "Output JSON array:"
    )

    raw = generate_text(
        prompt,
        system=system,
        model=model or DEFAULT_CHAT_MODEL,
        options={"temperature": 0.0, "top_p": 0.9, "max_tokens": DEFAULT_BATCH_TRANSLATE_MAX_TOKENS},
        base_url=base_url,
        timeout=timeout,
        api_key=api_key,
    )
    value = _strip_code_fences(raw)
    parsed = json.loads(value)
    if not isinstance(parsed, list) or len(parsed) != len(texts):
        raise ValueError("Batch translation returned invalid shape")
    return [str(item) for item in parsed]


def translate_html_preserving_tags(
    html: str,
    target_language: str,
    source_language: str = "auto",
    model=None,
    base_url=None,
    timeout=DEFAULT_TRANSLATE_TIMEOUT,
    api_key=None,
) -> str:
    payload = html or ""
    if not payload.strip():
        return payload

    parts = re.split(r"(<[^>]+>)", payload)
    translatable_idx = []
    raw_texts = []

    for i, part in enumerate(parts):
        if not part or part.startswith("<"):
            continue
        if not any(ch.isalpha() for ch in part):
            continue
        stripped = part.strip()
        if not stripped:
            continue
        translatable_idx.append(i)
        raw_texts.append(stripped)

    if not raw_texts:
        return payload

    unique_texts = []
    seen = set()
    for item in raw_texts:
        if item in seen:
            continue
        seen.add(item)
        unique_texts.append(item)

    translated_map = {}
    try:
        translated_unique = _batch_translate_texts(
            unique_texts,
            target_language=target_language,
            source_language=source_language,
            model=model,
            base_url=base_url,
            timeout=timeout,
            api_key=api_key,
        )
        translated_map = dict(zip(unique_texts, translated_unique))
    except Exception:
        # Batch tarjima muvaffaqiyatsiz bo'lsa, asl HTMLni buzmasdan fallback qilamiz.
        for item in unique_texts:
            try:
                translated_map[item] = translate_text(
                    item,
                    target_language=target_language,
                    source_language=source_language,
                    model=model,
                    base_url=base_url,
                    timeout=timeout,
                    api_key=api_key,
                )
            except Exception:
                translated_map[item] = item

    for idx in translatable_idx:
        original = parts[idx]
        stripped = original.strip()
        translated = translated_map.get(stripped, stripped)
        left_ws = len(original) - len(original.lstrip())
        right_ws = len(original) - len(original.rstrip())
        parts[idx] = f"{original[:left_ws]}{translated}{original[len(original) - right_ws:]}"

    return "".join(parts)


def translate_en_to_uz(text: str, **kwargs) -> str:
    return translate_text(text, target_language="uz", source_language="en", **kwargs)


def translate_uz_to_en(text: str, **kwargs) -> str:
    return translate_text(text, target_language="en", source_language="uz", **kwargs)


def adapt_message_for_search(message: str):
    language = detect_query_language(message)
    if language in {"en", "ru", "ko"}:
        return _translate_to_uz_for_search(message, source_language=language), language
    return message, language


def translate_query_for_search(message: str, source_language: str) -> str:
    if source_language in {"en", "ru", "ko"}:
        return _translate_to_uz_for_search(message, source_language=source_language)
    return message


def adapt_answer_for_user(answer: str, language: str) -> str:
    if language in _SUPPORTED_LANGS and language != "uz":
        return translate_text(answer, target_language=language, source_language="uz")
    return answer


def _looks_like_target_language(text: str, target_language: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    hangul, cyrillic, ascii_count, total = _script_counts(value)
    if target_language == "ko":
        return hangul > 0
    if target_language == "ru":
        return cyrillic > 0
    if target_language == "en":
        if cyrillic > 0 or hangul > 0:
            return False
        return _is_strong_english_text(value)
    if target_language == "uz":
        if cyrillic > 0 or hangul > 0:
            return False
        tokens = re.findall(r"[a-zA-Z']+", value.lower())
        uz_hits = sum(1 for t in tokens if t in _UZ_HINT_WORDS)
        en_hits = sum(1 for t in tokens if t in _EN_HINT_WORDS)
        if any(ch in value for ch in ("o'", "g'", "o‘", "g‘", "ʻ")):
            return True
        if uz_hits >= 2:
            return True
        if en_hits >= 2 and uz_hits == 0:
            return False
        return uz_hits >= en_hits
    return False


def ensure_answer_language(answer: str, target_language: str) -> str:
    if target_language not in _SUPPORTED_LANGS:
        return answer
    if _looks_like_target_language(answer, target_language):
        return answer

    # Avval aniq manba (uz), keyin auto bilan urinib ko'ramiz.
    last_candidate = answer
    for source in ("uz", "auto"):
        try:
            candidate = translate_text(answer, target_language=target_language, source_language=source)
        except Exception:
            continue
        if candidate and candidate.strip():
            last_candidate = candidate.strip()
            if _looks_like_target_language(last_candidate, target_language):
                return last_candidate
    return last_candidate
