import os
import time
import uuid
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

COLLECTION_NAME = "findoc_all"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_SIZE = 384

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ── Embedding ──────────────────────────────────────────────
def get_embeddings(texts: list[str]) -> list[list[float]]:
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    url = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction"
    response = requests.post(url, headers=headers, json={"inputs": texts})
    response.raise_for_status()
    result = response.json()
    return result

# ── Qdrant ─────────────────────────────────────────────────

def ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"[Retrieval] Created collection: {COLLECTION_NAME}")
    else:
        print(f"[Retrieval] Collection exists: {COLLECTION_NAME}")

def upload_documents(docs: list[Document]):
    ensure_collection()
    print(f"[Retrieval] Embedding {len(docs)} documents...")
    texts = [doc.page_content for doc in docs]
    embeddings = get_embeddings(texts)
    points = []
    for doc, vector in zip(docs, embeddings):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "page_content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "element_type": doc.metadata.get("element_type", ""),
                "original": doc.metadata.get("original", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
            }
        ))
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[Retrieval] Uploaded {len(points)} points to Qdrant")

# ── BM25 ───────────────────────────────────────────────────

def build_bm25_index(docs: list[Document]):
    tokenized = [doc.page_content.lower().split() for doc in docs]
    return BM25Okapi(tokenized), docs

def bm25_search(query: str, bm25, docs: list[Document], top_k: int = 10) -> list[Document]:
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [docs[i] for i in top_indices]

# ── Vector Search ──────────────────────────────────────────

def vector_search(query: str, top_k: int = 10) -> list[Document]:
    query_vector = get_embeddings([query])[0]
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points
    docs = []
    for r in results:
        docs.append(Document(
            page_content=r.payload["page_content"],
            metadata={
                "source": r.payload["source"],
                "element_type": r.payload["element_type"],
                "original": r.payload["original"],
                "chunk_index": r.payload["chunk_index"],
                "score": r.score,
            }
        ))
    return docs

# ── RRF Fusion ─────────────────────────────────────────────

def reciprocal_rank_fusion(vector_docs: list[Document], bm25_docs: list[Document], k: int = 60) -> list[Document]:
    scores = {}
    doc_map = {}

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]

# ── Reranker ───────────────────────────────────────────────

def rerank(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    query_vector = get_embeddings([query])[0]
    doc_vectors = get_embeddings([doc.page_content for doc in docs])
    
    def cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0

    scored = [(doc, cosine_similarity(query_vector, vec)) for doc, vec in zip(docs, doc_vectors)]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

# ── Main Retrieval Pipeline ────────────────────────────────

def retrieve(query: str, bm25, docs: list[Document], top_k: int = 5) -> list[Document]:
    print(f"[Retrieval] Query: {query}")
    v_docs = vector_search(query, top_k=10)
    b_docs = bm25_search(query, bm25, docs, top_k=10)
    fused = reciprocal_rank_fusion(v_docs, b_docs)
    print(f"[Retrieval] Fused {len(fused)} candidates, reranking...")
    reranked = rerank(query, fused, top_k=top_k)
    print(f"[Retrieval] Returning top {len(reranked)} results")
    return reranked


if __name__ == "__main__":
    import sys

    # ── Check if collection already has data ──
    count = client.count(collection_name=COLLECTION_NAME).count if COLLECTION_NAME in [c.name for c in client.get_collections().collections] else 0

    if count == 0:
        print("[Retrieval] No data in Qdrant, running ingestion first...")
        from ingestion import ingest_document
        if len(sys.argv) < 2:
            print("Usage: python retrieval.py <path_to_pdf>")
            sys.exit(1)
        docs = ingest_document(sys.argv[1])
        upload_documents(docs)
    else:
        print(f"[Retrieval] Found {count} points in Qdrant, skipping ingestion")
        # Load docs from Qdrant for BM25
        results = client.scroll(collection_name=COLLECTION_NAME, limit=500, with_payload=True)
        docs = [Document(
            page_content=r.payload["page_content"],
            metadata={
                "source": r.payload["source"],
                "element_type": r.payload["element_type"],
                "original": r.payload["original"],
                "chunk_index": r.payload["chunk_index"],
            }
        ) for r in results[0]]

    bm25, all_docs = build_bm25_index(docs)
    query = "What is GAAP and why is it important?"
    results = retrieve(query, bm25, all_docs)

    print(f"\n── Top Results ──")
    for i, doc in enumerate(results):
        print(f"\n[{i+1}] Source: {doc.metadata['source']}")
        print(f"    Summary : {doc.page_content[:200]}")
        print(f"    Original: {doc.metadata['original'][:200]}")