"""Step 1-4: load -> chunk -> embed -> index (Chroma, persisted to disk).

Usage:
  python ingest.py --hf --limit 3200      # rag-mini-wikipedia passages
  python ingest.py --pdfs my_docs/        # your own PDFs (one record per page)
  python ingest.py --hf --pdfs my_docs/   # both
"""
import argparse
import pathlib

import chromadb
from sentence_transformers import SentenceTransformer

import config


def load_hf(limit: int | None):
    from datasets import load_dataset
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    for row in ds:
        # each passage is one "page" of a pseudo-document called wiki-<id>
        yield {"text": row["passage"], "source": f"wiki-{row['id']}", "page": 1}


def load_pdfs(folder: str):
    from pypdf import PdfReader
    for pdf in sorted(pathlib.Path(folder).glob("**/*.pdf")):
        reader = PdfReader(str(pdf))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                yield {"text": text, "source": pdf.name, "page": i}


def chunk(text: str, tokenizer, size: int, overlap: int) -> list[str]:
    """Token-based sliding window so 'size' really means model tokens.
    We slice the ORIGINAL text via character offsets, so case and spacing survive."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if not offsets:
        return []
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(offsets), step):
        window = offsets[start:start + size]
        out.append(text[window[0][0]:window[-1][1]].strip())
        if start + size >= len(offsets):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="store_true", help="index rag-mini-wikipedia")
    ap.add_argument("--limit", type=int, default=None, help="max HF passages")
    ap.add_argument("--pdfs", default=None, help="folder with your PDFs")
    ap.add_argument("--reset", action="store_true", help="drop the collection first")
    args = ap.parse_args()
    if not (args.hf or args.pdfs):
        ap.error("pass --hf and/or --pdfs FOLDER")

    model = SentenceTransformer(config.EMBED_MODEL)
    tok = model.tokenizer

    client = chromadb.PersistentClient(path=config.DB_DIR)
    if args.reset:
        try:
            client.delete_collection(config.COLLECTION)
        except Exception:
            pass
    col = client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})

    records = []
    if args.hf:
        records += list(load_hf(args.limit))
    if args.pdfs:
        records += list(load_pdfs(args.pdfs))
    print(f"loaded {len(records)} pages/passages")

    ids, docs, metas = [], [], []
    for r in records:
        for j, c in enumerate(chunk(r["text"], tok, config.CHUNK_TOKENS, config.CHUNK_OVERLAP)):
            ids.append(f"{r['source']}::p{r['page']}::c{j}")
            docs.append(c)
            metas.append({"source": r["source"], "page": r["page"], "chunk": j})
    print(f"made {len(docs)} chunks (size={config.CHUNK_TOKENS}, overlap={config.CHUNK_OVERLAP})")

    B = 256
    for i in range(0, len(docs), B):
        vecs = model.encode(docs[i:i + B], normalize_embeddings=True, show_progress_bar=False)
        col.upsert(ids=ids[i:i + B], documents=docs[i:i + B],
                   metadatas=metas[i:i + B], embeddings=vecs.tolist())
        print(f"  indexed {min(i + B, len(docs))}/{len(docs)}")
    print(f"done. collection '{config.COLLECTION}' has {col.count()} chunks in ./{config.DB_DIR}")


if __name__ == "__main__":
    main()
