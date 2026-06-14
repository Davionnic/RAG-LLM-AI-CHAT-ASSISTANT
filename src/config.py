"""Application settings loaded from environment / .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_DIR", "chroma_db")
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")

COLLECTION_NAME = "documents"
