"""Alternative index: FAISS in memory (no server, very fast for experiments).
Reads the chunks already stored in Chroma, so run ingest.py first.

  python faiss_store.py "Was Abraham Lincoln the sixteenth President?"
"""
import sys

import chromadb
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config

col = chromadb.PersistentClient(path=config.DB_DIR).get_collection(config.COLLECTION)
data = col.get(include=["documents", "metadatas", "embeddings"])
vecs = np.asarray(data["embeddings"], dtype="float32")   # already L2-normalised

index = faiss.IndexFlatIP(vecs.shape[1])   # inner product = cosine on normalised vectors
index.add(vecs)

model = SentenceTransformer(config.EMBED_MODEL)
q = model.encode([sys.argv[1]], normalize_embeddings=True).astype("float32")
scores, idx = index.search(q, config.TOP_K)
for rank, (s, i) in enumerate(zip(scores[0], idx[0]), start=1):
    m = data["metadatas"][i]
    print(f"[{rank}] {m['source']} p.{m['page']}  cos={s:.3f}  {data['documents'][i][:90]!r}")
