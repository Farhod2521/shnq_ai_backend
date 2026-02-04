import hashlib
import math
import re

from .models import Clause, ClauseEmbedding
from .ollama_client import DEFAULT_EMBED_MODEL, embed_text as ollama_embed_text


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def embed_text(text, dim=256):
    vec = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % dim
        vec[idx] += 1.0

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a, b):
    return sum(x * y for x, y in zip(a, b))


def upsert_clause_embeddings(embedding_model=None, force_update=False, limit=None):
    model_name = embedding_model or DEFAULT_EMBED_MODEL
    qs = Clause.objects.select_related("document", "chapter").order_by("id")
    if limit:
        qs = qs[:limit]

    created = 0
    updated = 0
    skipped = 0

    for clause in qs:
        existing = ClauseEmbedding.objects.filter(clause=clause).first()

        if existing and not force_update and existing.embedding_model == model_name:
            skipped += 1
            continue

        vector = ollama_embed_text(clause.text, model=model_name)
        token_count = len(clause.text.split())

        _, was_created = ClauseEmbedding.objects.update_or_create(
            clause=clause,
            defaults={
                "embedding_model": model_name,
                "vector": vector,
                "token_count": token_count,
                "shnq_code": clause.document.code,
                "chapter_title": clause.chapter.title if clause.chapter else None,
                "clause_number": clause.clause_number,
                "lex_url": clause.document.lex_url,
            },
        )
        if was_created:
            created += 1
        else:
            updated += 1

    return {"created": created, "updated": updated, "skipped": skipped}
