"""Shared settings for the RAG project. Change values here, not in every file."""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current folder (OPENROUTER_API_KEY=...)

# all-MiniLM-L6-v2 only embeds the first 256 word-pieces of a chunk.
# bge-small-en-v1.5 embeds up to 512, so it fits 300-500 token chunks.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))
# Cosine distance in Chroma = 1 - cosine similarity. Above this = "not relevant".
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.6"))

DB_DIR = os.getenv("DB_DIR", "chroma_db")
COLLECTION = os.getenv("COLLECTION", "docs")

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")  # any id from openrouter.ai/models
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
