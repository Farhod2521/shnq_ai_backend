import hashlib
import math
import os
import re
from typing import Any

from openai import OpenAI


DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_EMBED_MODEL = os.getenv("DEEPSEEK_EMBED_MODEL", "deepseek-embedding")
DEFAULT_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
DEFAULT_EMBED_DIM = int(os.getenv("DEEPSEEK_EMBED_DIM", "768"))
DEFAULT_CHAT_TIMEOUT = int(os.getenv("DEEPSEEK_CHAT_TIMEOUT", "45"))
DEFAULT_EMBED_TIMEOUT = int(os.getenv("DEEPSEEK_EMBED_TIMEOUT", "12"))
EMBEDDING_FALLBACK = os.getenv("DEEPSEEK_EMBED_FALLBACK", "semantic_hash")
STRICT_EMBEDDING = os.getenv("DEEPSEEK_EMBED_STRICT", "0") == "1"

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_EMBED_MODEL_SUPPORT = {}


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
