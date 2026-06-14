"""Load documents, split them into chunks, embed, and persist to Chroma.

Run from the project root:
    python -m src.ingest            # ingest everything in DATA_DIR
    python -m src.ingest a.pdf b.md # ingest specific files
"""

import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.llm import get_embeddings

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
    ".docx": Docx2txtLoader,
}

SUPPORTED_EXTENSIONS = set(LOADERS)


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def load_documents(paths: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for path in paths:
        loader_factory = LOADERS.get(path.suffix.lower())
        if loader_factory is None:
            print(f"Skipping unsupported file: {path.name}")
            continue
        loaded = loader_factory(str(path)).load()
        for doc in loaded:
            doc.metadata["source"] = path.name
        docs.extend(loaded)
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)


def ingest_paths(paths: list[Path]) -> int:
    """Ingest the given files into the vector store. Returns chunk count added."""
    docs = load_documents(paths)
    if not docs:
        return 0
    chunks = split_documents(docs)
    get_vector_store().add_documents(chunks)
    return len(chunks)


def ingest_directory() -> int:
    """Ingest every supported file in DATA_DIR. Returns chunk count added."""
    paths = [
        p
        for p in sorted(config.DATA_DIR.glob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not paths:
        print(f"No supported documents found in {config.DATA_DIR}")
        return 0
    print(f"Ingesting {len(paths)} file(s): {', '.join(p.name for p in paths)}")
    return ingest_paths(paths)


def index_size() -> int:
    """Number of chunks currently in the vector store (0 if none)."""
    if not config.CHROMA_DIR.exists():
        return 0
    try:
        return get_vector_store()._collection.count()
    except Exception:
        return 0


def main() -> None:
    args = sys.argv[1:]
    if args:
        count = ingest_paths([Path(a) for a in args])
    else:
        count = ingest_directory()
    print(f"Done. Added {count} chunk(s). Index now holds {index_size()} chunk(s).")


if __name__ == "__main__":
    main()
