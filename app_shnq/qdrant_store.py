import os
import re

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception as exc:  # pragma: no cover - import guard
    QdrantClient = None
    qmodels = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "clause_embeddings").strip()

_client = None
_COLLECTION_READY = False
_COLLECTION_VECTOR_SIZE = None


def _require_client():
    if QdrantClient is None:
        raise RuntimeError(
            "qdrant-client o'rnatilmagan yoki import qilinmadi: " + str(_IMPORT_ERROR)
        )


def client():
    _require_client()
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def normalize_doc_code(code: str) -> str:
    return re.sub(r"\s+", "", (code or "")).lower()


def doc_code_prefixes(code: str):
    norm = normalize_doc_code(code)
    if not norm:
        return []
    parts = re.split(r"([.-])", norm)
    acc = ""
    prefixes = []
    for part in parts:
        acc += part
        if part in ".-":
            continue
        prefixes.append(acc)

    seen = set()
    ordered = []
    for item in prefixes:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _collection_exists():
    try:
        names = [c.name for c in client().get_collections().collections]
    except Exception:
        return False
    return QDRANT_COLLECTION in names

def ensure_collection(vector_size: int, force_create: bool = False):
    global _COLLECTION_READY, _COLLECTION_VECTOR_SIZE
    if _COLLECTION_READY:
        if _COLLECTION_VECTOR_SIZE != vector_size:
            raise ValueError(
                "Qdrant collection vector size mismatch: expected "
                + str(_COLLECTION_VECTOR_SIZE)
                + ", got "
                + str(vector_size)
            )
        return

    if force_create:
        try:
            client().create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )
        except Exception:
            pass
        try:
            info = client().get_collection(QDRANT_COLLECTION)
            existing_size = info.config.params.vectors.size
        except Exception:
            _COLLECTION_READY = True
            _COLLECTION_VECTOR_SIZE = vector_size
            return
        if existing_size != vector_size:
            raise ValueError(
                "Qdrant collection vector size mismatch: expected "
                + str(existing_size)
                + ", got "
                + str(vector_size)
            )
        _COLLECTION_READY = True
        _COLLECTION_VECTOR_SIZE = existing_size
        return

    if not _collection_exists():
        client().create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )
        _COLLECTION_READY = True
        _COLLECTION_VECTOR_SIZE = vector_size
        return

    info = client().get_collection(QDRANT_COLLECTION)
    existing_size = info.config.params.vectors.size
    if existing_size != vector_size:
        raise ValueError(
            "Qdrant collection vector size mismatch: expected "
            + str(existing_size)
            + ", got "
            + str(vector_size)
        )
    _COLLECTION_READY = True
    _COLLECTION_VECTOR_SIZE = existing_size

def count_points():
    if not _collection_exists():
        return 0
    try:
        result = client().count(collection_name=QDRANT_COLLECTION, exact=True)
        return result.count
    except Exception as exc:
        msg = str(exc)
        if "doesn't exist" in msg or "Not found" in msg:
            return 0
        raise


def health_check():
    try:
        _require_client()
        info = client().get_collections()
        names = [c.name for c in getattr(info, "collections", [])] if info else []
        return {
            "ok": True,
            "collection_exists": QDRANT_COLLECTION in names,
            "collection": QDRANT_COLLECTION,
            "url": QDRANT_URL,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "collection": QDRANT_COLLECTION,
            "url": QDRANT_URL,
        }

def upsert_point(point_id, vector, payload):
    if not vector:
        return
    ensure_collection(len(vector))
    try:
        client().upsert(
            collection_name=QDRANT_COLLECTION,
            points=[qmodels.PointStruct(id=point_id, vector=vector, payload=payload)],
        )
    except Exception as exc:
        msg = str(exc)
        if "doesn't exist" in msg or "Not found" in msg:
            ensure_collection(len(vector), force_create=True)
            client().upsert(
                collection_name=QDRANT_COLLECTION,
                points=[qmodels.PointStruct(id=point_id, vector=vector, payload=payload)],
            )
            return
        raise

def _query_points(client_instance, vector, limit, query_filter):
    if hasattr(client_instance, "query_points"):
        try:
            res = client_instance.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vector,
                limit=limit,
                query_filter=query_filter,
            )
        except TypeError:
            res = client_instance.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vector,
                limit=limit,
                filter=query_filter,
            )
        return getattr(res, "points", res)

    if hasattr(client_instance, "search"):
        return client_instance.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
        )

    if hasattr(client_instance, "search_points"):
        return client_instance.search_points(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
        )

    raise RuntimeError("Qdrant client query/search method topilmadi")


def search(vector, limit=20, doc_code=None):
    if not vector:
        return []
    ensure_collection(len(vector))

    query_filter = None
    if doc_code:
        norm = normalize_doc_code(doc_code)
        if norm:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="shnq_code_prefixes",
                        match=qmodels.MatchValue(value=norm),
                    )
                ]
            )

    return _query_points(client(), vector, limit, query_filter)
