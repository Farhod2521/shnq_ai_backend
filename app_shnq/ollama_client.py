import json
import os
import urllib.error
import urllib.request


DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")


def _post_json(path, payload, base_url=None, timeout=60):
    base = (base_url or DEFAULT_BASE_URL).rstrip("/")
    url = f"{base}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
    return json.loads(body.decode("utf-8"))


def embed_text(text, model=None, base_url=None, timeout=60):
    if not text:
        return []
    payload = {
        "model": model or DEFAULT_EMBED_MODEL,
        "prompt": text,
    }
    data = _post_json("/api/embeddings", payload, base_url=base_url, timeout=timeout)
    embedding = data.get("embedding")
    if not embedding:
        raise ValueError("Ollama embedding javobi bo'sh.")
    return embedding


def generate_text(prompt, system=None, model=None, options=None, base_url=None, timeout=120):
    payload = {
        "model": model or DEFAULT_CHAT_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if options:
        payload["options"] = options
    data = _post_json("/api/generate", payload, base_url=base_url, timeout=timeout)
    return (data.get("response") or "").strip()
