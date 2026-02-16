import os

from .models import Clause, ClauseEmbedding
from .deepseek_client import DEFAULT_EMBED_MODEL, embed_text as deepseek_embed_text
from .qdrant_store import doc_code_prefixes, normalize_doc_code, upsert_point


USE_QDRANT = os.getenv("RAG_USE_QDRANT", "0") == "1"


def cosine_similarity(a, b):
    return sum(x * y for x, y in zip(a, b))


def _build_qdrant_payload(clause: Clause):
    shnq_code = clause.document.code
    return {
        "shnq_code": shnq_code,
        "shnq_code_norm": normalize_doc_code(shnq_code),
        "shnq_code_prefixes": doc_code_prefixes(shnq_code),
        "chapter_title": clause.chapter.title if clause.chapter else None,
        "clause_number": clause.clause_number,
        "lex_url": clause.document.lex_url,
    }


def _upsert_qdrant(clause: Clause, vector):
    if not USE_QDRANT:
        return
    if not vector:
        return
    payload = _build_qdrant_payload(clause)
    upsert_point(str(clause.id), vector, payload)


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
            if USE_QDRANT and existing.vector:
                _upsert_qdrant(clause, existing.vector)
            continue

        vector = deepseek_embed_text(clause.text, model=model_name)
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
        _upsert_qdrant(clause, vector)
        if was_created:
            created += 1
        else:
            updated += 1

    return {"created": created, "updated": updated, "skipped": skipped}
