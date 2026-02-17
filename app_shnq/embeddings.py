import os

from .models import (
    Clause,
    ClauseEmbedding,
    ImageEmbedding,
    NormImage,
    ensure_runtime_tables,
)
from .deepseek_client import DEFAULT_EMBED_MODEL, embed_text as deepseek_embed_text
from .qdrant_store import doc_code_prefixes, normalize_doc_code, upsert_point


USE_QDRANT = os.getenv("RAG_USE_QDRANT", "0") == "1"


def cosine_similarity(a, b):
    return sum(x * y for x, y in zip(a, b))


def _build_qdrant_payload(clause: Clause):
    shnq_code = clause.document.code
    return {
        "source_type": "clause",
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


def _build_image_embedding_text(image: NormImage) -> str:
    parts = [f"Hujjat: {image.document.code}"]
    if image.section_title:
        parts.append(f"Bolim: {image.section_title}")
    elif image.chapter:
        parts.append(f"Bob: {image.chapter.title}")
    if image.appendix_number:
        parts.append(f"Ilova: {image.appendix_number}")
    if image.title:
        parts.append(f"Sarlavha: {image.title}")
    if image.context_text:
        parts.append(f"Kontekst: {image.context_text}")
    if image.ocr_text:
        parts.append(f"OCR: {image.ocr_text}")
    parts.append(f"Rasm URL: {image.image_url}")
    return "\n".join(parts)


def upsert_clause_embeddings(embedding_model=None, force_update=False, limit=None):
    ensure_runtime_tables()
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


def upsert_image_embeddings(embedding_model=None, force_update=False, limit=None):
    ensure_runtime_tables()
    model_name = embedding_model or DEFAULT_EMBED_MODEL
    qs = NormImage.objects.select_related("document", "chapter").order_by("id")
    if limit:
        qs = qs[:limit]

    created = 0
    updated = 0
    skipped = 0

    for image in qs:
        existing = ImageEmbedding.objects.filter(image=image).first()
        if existing and not force_update and existing.embedding_model == model_name:
            skipped += 1
            continue

        text_for_embedding = _build_image_embedding_text(image)
        if not text_for_embedding.strip():
            skipped += 1
            continue

        vector = deepseek_embed_text(text_for_embedding, model=model_name)
        token_count = len(text_for_embedding.split())

        _, was_created = ImageEmbedding.objects.update_or_create(
            image=image,
            defaults={
                "embedding_model": model_name,
                "vector": vector,
                "token_count": token_count,
                "shnq_code": image.document.code,
                "chapter_title": image.section_title or (image.chapter.title if image.chapter else None),
                "appendix_number": image.appendix_number,
                "image_url": image.image_url,
            },
        )
        if was_created:
            created += 1
        else:
            updated += 1

    return {"created": created, "updated": updated, "skipped": skipped}


def upsert_all_embeddings(embedding_model=None, force_update=False, limit=None):
    clause_result = upsert_clause_embeddings(
        embedding_model=embedding_model,
        force_update=force_update,
        limit=limit,
    )
    image_result = upsert_image_embeddings(
        embedding_model=embedding_model,
        force_update=force_update,
        limit=limit,
    )
    return {"clauses": clause_result, "images": image_result}
