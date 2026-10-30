"""Step 5-6: retrieve top-k chunks and answer with numbered citations.

Usage:
  python rag.py "Who was the sixteenth president of the United States?"
  python rag.py --no-llm "question"     # retrieval only, no API key needed
"""
import argparse
import os
from functools import lru_cache

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import config

IDK = "I don't know based on the provided documents."

SYSTEM = (
    "You answer questions using ONLY the numbered context passages. "
    "After every sentence that uses a passage, cite it like [1] or [2][3]. "
    f"If the passages do not contain the answer, reply exactly: {IDK} "
    "Never use outside knowledge. Be concise."
)


@lru_cache
def _model():
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache
def _collection():
    client = chromadb.PersistentClient(path=config.DB_DIR)
    return client.get_collection(config.COLLECTION)


@lru_cache
def _llm():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY missing: put it in .env")
    return OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=key)


def retrieve(question: str, k: int = config.TOP_K) -> list[dict]:
    q = _model().encode([question], normalize_embeddings=True).tolist()
    res = _collection().query(query_embeddings=q, n_results=k,
                              include=["documents", "metadatas", "distances"])
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist})
    return hits


def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = [f"[{i}] (source: {h['source']}, page {h['page']})\n{h['text']}"
              for i, h in enumerate(hits, start=1)]
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def answer(question: str, k: int = config.TOP_K, use_llm: bool = True) -> dict:
    hits = [h for h in retrieve(question, k) if h["distance"] <= config.MAX_DISTANCE]
    if not hits:  # guard 1: nothing relevant retrieved -> refuse without calling the LLM
        return {"answer": IDK, "sources": []}
    if not use_llm:
        return {"answer": "(LLM skipped)", "sources": hits}
    resp = _llm().chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": build_prompt(question, hits)}],
    )
    text = resp.choices[0].message.content.strip()  # guard 2: the system prompt
    return {"answer": text, "sources": hits, "usage": resp.usage}


def format_sources(hits: list[dict]) -> str:
    return "\n".join(f"  [{i}] {h['source']} p.{h['page']}  (distance {h['distance']:.3f})"
                     for i, h in enumerate(hits, start=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    out = answer(a.question, use_llm=not a.no_llm)
    print(out["answer"])
    if out["sources"]:
        print("\nSources:\n" + format_sources(out["sources"]))
