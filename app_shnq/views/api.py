import logging
import re
import time

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from ..embeddings import cosine_similarity
from ..deepseek_client import (
    detect_query_language,
    ensure_answer_language,
    embed_text,
    generate_text,
    translate_html_preserving_tags,
    translate_query_for_search,
)
from ..models import QuestionAnswer
from ..table_i18n import (
    ensure_normtable_i18n_columns,
    get_pretranslated_table_content,
    has_pretranslated_table_content,
)
from .constants import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    KEYWORD_WEIGHT,
    MIN_SCORE,
    RAG_FINAL_MAX_TOKENS,
    RAG_IMAGE_TOP_K,
    RAG_TABLE_ROW_TOP_K,
    RAG_MULTILINGUAL_NATIVE_FIRST,
    RAG_MULTILINGUAL_TRANSLATE_FALLBACK,
    RAG_TRANSLATED_QUERY_SCORE_WEIGHT,
    RAG_TRANSLATION_FALLBACK_THRESHOLD,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    STRICT_MIN_SCORE,
    USE_QDRANT,
)
from .services import (
    _build_greeting_response,
    _build_out_of_scope_response,
    _build_document_clarification_answer,
    _build_rag_prompt,
    _build_table_answer,
    _build_table_qa_answer,
    _can_answer_with_relaxed_threshold,
    _cleanup_answer_format,
    _ensure_embeddings,
    _extract_image_url_from_text,
    _extract_doc_code,
    _extract_query_terms,
    _filter_embeddings_by_doc_code,
    _find_table_for_query,
    _get_embeddings_for_query,
    _is_clearly_out_of_scope,
    _is_greeting,
    _is_image_clause_text,
    _is_shnq_related,
    _is_table_direct_lookup_request,
    _is_table_request,
    _keyword_score,
    _llm_rerank,
    _linked_image_embeddings_from_clause_refs,
    _needs_clarification,
    _pick_related_table_from_rag,
    _pick_related_table_from_row_hits,
    _pick_fewshot_examples,
    _rewrite_query_if_needed,
    _search_image_embeddings,
    _search_table_row_embeddings_global,
    _score_with_qdrant,
    _should_ask_document_clarification,
    _table_candidate_chapters,
    _table_candidate_docs,
    _image_text_for_context,
)

logger = logging.getLogger(__name__)
CLAUSE_LOOKUP_RE = re.compile(
    r"(?:\b\d+\s*[-.]?\s*band(?:da|ni|ga|dan|ning|lar)?\b|\bband(?:da|ni|ga|dan|ning|lar)?\b|\bmodda\b)",
    re.IGNORECASE,
)


def _is_explicit_clause_lookup(text: str) -> bool:
    return bool(CLAUSE_LOOKUP_RE.search(text or ""))


def _merge_scored_candidates(primary, secondary, secondary_weight=1.0):
    merged = {}

    for score, emb, semantic, keyword in primary:
        merged[str(emb.clause_id)] = (score, emb, semantic, keyword)

    for score, emb, semantic, keyword in secondary:
        weighted_score = score * secondary_weight
        current = merged.get(str(emb.clause_id))
        candidate = (weighted_score, emb, semantic, keyword)
        if current is None or weighted_score > current[0]:
            merged[str(emb.clause_id)] = candidate

    combined = list(merged.values())
    combined.sort(key=lambda x: x[0], reverse=True)
    return combined


def _merge_image_candidates(primary, secondary, secondary_weight=1.0):
    merged = {}

    for score, emb, semantic, keyword in primary:
        merged[str(emb.image_id)] = (score, emb, semantic, keyword)

    for score, emb, semantic, keyword in secondary:
        weighted_score = score * secondary_weight
        current = merged.get(str(emb.image_id))
        candidate = (weighted_score, emb, semantic, keyword)
        if current is None or weighted_score > current[0]:
            merged[str(emb.image_id)] = candidate

    combined = list(merged.values())
    combined.sort(key=lambda x: x[0], reverse=True)
    return combined


def _merge_table_row_candidates(primary, secondary, secondary_weight=1.0):
    merged = {}

    for score, emb, semantic, keyword in primary:
        merged[str(emb.row_id)] = (score, emb, semantic, keyword)

    for score, emb, semantic, keyword in secondary:
        weighted_score = score * secondary_weight
        current = merged.get(str(emb.row_id))
        candidate = (weighted_score, emb, semantic, keyword)
        if current is None or weighted_score > current[0]:
            merged[str(emb.row_id)] = candidate

    combined = list(merged.values())
    combined.sort(key=lambda x: x[0], reverse=True)
    return combined


def _build_retrieval_fallback_answer(top_pairs, image_pairs=None, table_row_pairs=None):
    if not top_pairs:
        if image_pairs:
            return "Mos rasm topildi. Rasm URL manbalar bo'limida berildi."
        if table_row_pairs:
            best_row_text = (table_row_pairs[0][1].search_text or "").strip()
            if best_row_text:
                return best_row_text
        return "Mos band topilmadi."

    _score, emb, _semantic, _keyword = top_pairs[0]
    clause_text = (emb.clause.text or "").strip()
    if not clause_text:
        return "LLM vaqtincha ishlamayapti, lekin mos manba topildi."
    if _is_image_clause_text(clause_text) and _extract_image_url_from_text(clause_text):
        return "Mos rasm topildi. Rasm URL manbalar bo'limida berildi."

    return clause_text


def _build_clause_source(score, emb, semantic, keyword):
    clause = emb.clause
    source = {
        "type": "clause",
        "shnq_code": emb.shnq_code,
        "chapter": emb.chapter_title,
        "clause_number": emb.clause_number,
        "html_anchor": clause.html_anchor,
        "lex_url": emb.lex_url,
        "snippet": (clause.text or "")[:280],
        "score": round(score, 4),
        "semantic_score": round(semantic, 4),
        "keyword_score": round(keyword, 4),
    }
    clause_text = clause.text or ""
    image_url = _extract_image_url_from_text(clause_text)
    if _is_image_clause_text(clause_text) and image_url:
        cleaned = clause_text.replace("[IMAGE]", "Rasm:", 1)
        if "URL:" in cleaned:
            cleaned = cleaned.split("URL:", 1)[0].rstrip(" |")
        source["type"] = "image"
        source["image_url"] = image_url
        source["snippet"] = cleaned[:280]
    return source


def _build_image_source(score, emb, semantic, keyword):
    image = emb.image
    return {
        "type": "image",
        "shnq_code": emb.shnq_code,
        "chapter": emb.chapter_title,
        "appendix_number": emb.appendix_number,
        "title": image.title,
        "html_anchor": image.html_anchor,
        "image_url": image.image_url,
        "snippet": _image_text_for_context(image)[:280],
        "score": round(score, 4),
        "semantic_score": round(semantic, 4),
        "keyword_score": round(keyword, 4),
    }


def _build_table_row_source(score, emb, semantic, keyword):
    row = emb.row
    table = row.table
    chapter_title = table.section_title or (table.chapter.title if table.chapter else None)
    snippet = (emb.search_text or "").strip()
    if len(snippet) > 320:
        snippet = snippet[:320].rsplit(" ", 1)[0].strip() + " ..."
    return {
        "type": "table_row",
        "shnq_code": emb.shnq_code,
        "chapter": chapter_title,
        "table_number": table.table_number,
        "title": table.title,
        "html_anchor": table.html_anchor,
        "row_index": emb.row_index,
        "snippet": snippet,
        "score": round(score, 4),
        "semantic_score": round(semantic, 4),
        "keyword_score": round(keyword, 4),
    }


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
            meta = response.data.get("meta") if isinstance(response.data, dict) else None
            table_prelocalized = isinstance(meta, dict) and meta.get("table_prelocalized") is True
            table_html = response.data.get("table_html")
            if (
                isinstance(table_html, str)
                and table_html.strip()
                and language in {"en", "ru", "ko"}
                and not table_prelocalized
            ):
                try:
                    response.data["table_html"] = translate_html_preserving_tags(
                        table_html,
                        target_language=language,
                        source_language="uz",
                    )
                except Exception:
                    pass
            sources = response.data.get("sources")
            if isinstance(sources, list) and language in {"en", "ru", "ko"} and not table_prelocalized:
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
        ensure_normtable_i18n_columns()
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
                    "markdown": get_pretranslated_table_content(table, message_language)[1],
                    "html": get_pretranslated_table_content(table, message_language)[0],
                }
            ]
            table_html, _table_md = get_pretranslated_table_content(table, message_language)
            QuestionAnswer.objects.create(
                question=original_message,
                answer=answer,
                top_clause_ids=[],
            )
            return Response(
                {
                    "answer": answer,
                    "sources": sources,
                    "table_html": table_html,
                    "meta": {
                        "type": "table_lookup",
                        "model": CHAT_MODEL,
                        "table_prelocalized": has_pretranslated_table_content(table, message_language),
                    },
                }
            )

        try:
            _ensure_embeddings()
        except Exception as exc:
            return Response({"error": f"Embedding tayyorlash xatoligi: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        requested_doc_code = _extract_doc_code(original_message) or _extract_doc_code(search_message)
        non_uz_query = message_language in {"en", "ru", "ko"}
        translated_search_message = search_message if non_uz_query else None

        primary_query_message = search_message
        secondary_query_message = None
        if non_uz_query and RAG_MULTILINGUAL_NATIVE_FIRST:
            primary_query_message = original_message
            if RAG_MULTILINGUAL_TRANSLATE_FALLBACK:
                secondary_query_message = translated_search_message

        rewritten_primary = _rewrite_query_if_needed(primary_query_message)
        rewritten_secondary = None
        translation_fallback_used = False
        embeddings = None
        scoped_doc_embeddings = None

        def compute_scored(query_text):
            nonlocal embeddings, scoped_doc_embeddings
            t_embed = time.perf_counter()
            query_vec = embed_text(query_text, model=EMBEDDING_MODEL)
            timings["embed"] = round(timings["embed"] + ((time.perf_counter() - t_embed) * 1000), 2)
            query_terms = _extract_query_terms(query_text)

            if embeddings is None:
                embeddings = _get_embeddings_for_query()
            scoped_embeddings = embeddings
            if requested_doc_code:
                if scoped_doc_embeddings is None:
                    scoped_doc_embeddings = _filter_embeddings_by_doc_code(embeddings, requested_doc_code)
                scoped_embeddings = scoped_doc_embeddings

            def local_score():
                local_scored = []
                for emb in scoped_embeddings:
                    semantic = cosine_similarity(query_vec, emb.vector)
                    keyword = _keyword_score(query_terms, emb)
                    score = semantic + (KEYWORD_WEIGHT * keyword)
                    local_scored.append((score, emb, semantic, keyword))
                local_scored.sort(key=lambda x: x[0], reverse=True)
                return local_scored

            if USE_QDRANT:
                qdrant_scored = _score_with_qdrant(query_vec, query_terms, requested_doc_code=requested_doc_code)
                # Qdrant eskirgan pointlar sabab kam natija qaytarsa, local indeks bilan to'ldiramiz.
                if len(qdrant_scored) >= 8:
                    return qdrant_scored
                fallback_local = local_score()
                if not qdrant_scored:
                    return fallback_local
                return _merge_scored_candidates(qdrant_scored, fallback_local, secondary_weight=1.0)

            return local_score()

        try:
            scored = compute_scored(rewritten_primary)
        except Exception as exc:
            error_name = "Qdrant" if "qdrant" in str(exc).lower() else "Embedding"
            return Response({"error": f"{error_name} xatoligi: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        primary_best_score = scored[0][0] if scored else 0.0
        can_try_translated_fallback = (
            secondary_query_message
            and secondary_query_message.strip()
            and secondary_query_message.strip().lower() != primary_query_message.strip().lower()
        )
        should_try_translated_fallback = (
            can_try_translated_fallback
            and ((not scored) or (primary_best_score < RAG_TRANSLATION_FALLBACK_THRESHOLD))
        )
        if should_try_translated_fallback:
            rewritten_secondary = _rewrite_query_if_needed(secondary_query_message)
            try:
                translated_scored = compute_scored(rewritten_secondary)
            except Exception as exc:
                logger.warning("translated_fallback_failed: %s", exc)
                translated_scored = []
            if translated_scored:
                scored = _merge_scored_candidates(
                    scored,
                    translated_scored,
                    secondary_weight=RAG_TRANSLATED_QUERY_SCORE_WEIGHT,
                )
                translation_fallback_used = True

        image_pairs = _search_image_embeddings(
            rewritten_primary,
            requested_doc_code=requested_doc_code,
        )
        if rewritten_secondary:
            secondary_image_pairs = _search_image_embeddings(
                rewritten_secondary,
                requested_doc_code=requested_doc_code,
            )
            if secondary_image_pairs:
                image_pairs = _merge_image_candidates(
                    image_pairs,
                    secondary_image_pairs,
                    secondary_weight=RAG_TRANSLATED_QUERY_SCORE_WEIGHT,
                )

        table_row_pairs = []
        explicit_clause_lookup = _is_explicit_clause_lookup(search_message)
        if not explicit_clause_lookup:
            table_row_pairs = _search_table_row_embeddings_global(
                rewritten_primary,
                requested_doc_code=requested_doc_code,
                limit=max(1, RAG_TABLE_ROW_TOP_K),
            )
            if rewritten_secondary:
                secondary_table_row_pairs = _search_table_row_embeddings_global(
                    rewritten_secondary,
                    requested_doc_code=requested_doc_code,
                    limit=max(1, RAG_TABLE_ROW_TOP_K),
                )
                if secondary_table_row_pairs:
                    table_row_pairs = _merge_table_row_candidates(
                        table_row_pairs,
                        secondary_table_row_pairs,
                        secondary_weight=RAG_TRANSLATED_QUERY_SCORE_WEIGHT,
                    )
            # Agar clause qidiruvi aniq va kuchli chiqsa, table-row kontekstni qo'shmaymiz.
            if table_row_pairs and scored:
                top_clause_score = scored[0][0]
                top_clause_keyword = scored[0][3]
                top_table_row_score = table_row_pairs[0][0]
                if top_clause_keyword >= 0.2 and (top_table_row_score + 0.03) < top_clause_score:
                    table_row_pairs = []

        if requested_doc_code and not scored and not image_pairs and not table_row_pairs:
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

        rewritten_any = (rewritten_primary != primary_query_message) or (
            rewritten_secondary is not None
            and secondary_query_message is not None
            and rewritten_secondary != secondary_query_message
        )
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
            if allow_relaxed or image_pairs or table_row_pairs:
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
                            "rewritten": rewritten_any,
                        },
                    }
                )

        filtered = [
            (score, item, semantic, keyword)
            for score, item, semantic, keyword in scored
            if score >= MIN_SCORE
        ]
        candidate_pairs = filtered[: max(RERANK_CANDIDATES, 5)]
        reranked = _llm_rerank(original_message, candidate_pairs)
        top_pairs = reranked[:5]
        top = [item for _, item, *_rest in top_pairs]
        image_limit = max(1, RAG_IMAGE_TOP_K)
        image_top_pairs = image_pairs[:image_limit]
        table_row_limit = max(1, RAG_TABLE_ROW_TOP_K)
        table_row_top_pairs = table_row_pairs[:table_row_limit]
        linked_image_pairs = _linked_image_embeddings_from_clause_refs(
            top_pairs,
            original_message,
            requested_doc_code=requested_doc_code,
        )
        if linked_image_pairs:
            image_limit = max(image_limit, len(linked_image_pairs))
            image_top_pairs = _merge_image_candidates(image_top_pairs, linked_image_pairs)[:image_limit]

        if not top and not image_top_pairs and not table_row_top_pairs:
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

        fewshot_examples = _pick_fewshot_examples(original_message, limit=3)
        system, prompt = _build_rag_prompt(
            original_message,
            top,
            response_language=message_language,
            fewshot_examples=fewshot_examples,
            image_sources=[item for _score, item, _semantic, _keyword in image_top_pairs],
            table_row_sources=[item for _score, item, _semantic, _keyword in table_row_top_pairs],
        )
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
            logger.warning("rag_generate_failed: %s", exc)
            answer = _build_retrieval_fallback_answer(
                top_pairs,
                image_pairs=image_top_pairs,
                table_row_pairs=table_row_top_pairs,
            )
        if not answer:
            if top:
                answer = top[0].clause.text
            elif image_top_pairs:
                answer = "Mos rasm topildi. Rasm URL manbalar bo'limida berildi."
            elif table_row_top_pairs:
                answer = (table_row_top_pairs[0][1].search_text or "").strip() or "Mos jadval satri topildi."
            else:
                answer = "Mos band topilmadi."
        answer = _cleanup_answer_format(answer, response_language=message_language)
        sources = []
        for score, emb, semantic, keyword in top_pairs:
            sources.append(_build_clause_source(score, emb, semantic, keyword))
        for score, emb, semantic, keyword in image_top_pairs:
            sources.append(_build_image_source(score, emb, semantic, keyword))
        for score, emb, semantic, keyword in table_row_top_pairs:
            sources.append(_build_table_row_source(score, emb, semantic, keyword))

        related_table = _pick_related_table_from_rag(search_message, top_pairs)
        if not related_table:
            related_table = _pick_related_table_from_row_hits(table_row_top_pairs)
        table_html = None
        if related_table:
            table_html, table_md = get_pretranslated_table_content(related_table, message_language)
            sources.append(
                {
                    "type": "table",
                    "shnq_code": related_table.document.code,
                    "chapter": related_table.section_title
                    or (related_table.chapter.title if related_table.chapter else None),
                    "table_number": related_table.table_number,
                    "title": related_table.title,
                    "html_anchor": related_table.html_anchor,
                    "markdown": table_md,
                    "html": table_html,
                }
            )
        image_urls = []
        seen_image_urls = set()
        for item in sources:
            if not isinstance(item, dict):
                continue
            url = item.get("image_url")
            if not url or url in seen_image_urls:
                continue
            seen_image_urls.add(url)
            image_urls.append(url)

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
                "image_urls": image_urls,
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
                    "rewritten": rewritten_any,
                    "query_used": rewritten_primary,
                    "query_used_fallback": rewritten_secondary,
                    "query_original": original_message,
                    "multilingual_native_first": RAG_MULTILINGUAL_NATIVE_FIRST,
                    "translation_fallback_used": translation_fallback_used,
                    "translation_fallback_threshold": RAG_TRANSLATION_FALLBACK_THRESHOLD,
                    "translated_query_score_weight": RAG_TRANSLATED_QUERY_SCORE_WEIGHT,
                    "fewshot_examples_used": len(fewshot_examples),
                    "image_sources": len(image_top_pairs),
                    "table_row_sources": len(table_row_top_pairs),
                    "table_prelocalized": (
                        has_pretranslated_table_content(related_table, message_language) if related_table else False
                    ),
                },
            }
        )



