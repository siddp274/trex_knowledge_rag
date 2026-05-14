"""
Document loaders for multiple file formats.

Thin wrappers around LangChain community loaders, producing a
standardized dict format that feeds into the chunking pipeline.

Each loaded document becomes:
{
    "id": SHA-512 hash of the text (deterministic),
    "title": filename or extracted title,
    "text": full document text,
    "source": file path or URL,
    "creation_date": ISO-8601 string or empty,
    "raw_data": any extra metadata from the loader
}
"""
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    BSHTMLLoader,
)

from ingestion.hashing import gen_sha512_hash


def load_single_source(source: str) -> list[dict[str, Any]]:
    """
    Load a single file or URL into standardized document dicts.
    
    Returns a list because some formats (PDF, CSV) produce multiple
    page/row documents that we then concatenate per-source in the
    main ingestion function.
    """
    source = source.strip()

    if source.startswith(("http://", "https://")):
        loader = WebBaseLoader(
            web_paths=[source],
            requests_kwargs={"verify": True, "timeout": 30},
        )
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        suffix = path.suffix.lower()
        loader_map = {
            ".pdf": lambda: PyPDFLoader(source),
            ".csv": lambda: CSVLoader(source),
            ".md": lambda: UnstructuredMarkdownLoader(source),
            ".markdown": lambda: UnstructuredMarkdownLoader(source),
            ".html": lambda: BSHTMLLoader(source),
            ".htm": lambda: BSHTMLLoader(source),
        }
        loader = loader_map.get(suffix, lambda: TextLoader(source, encoding="utf-8"))()

    lc_docs = loader.load()

    # Convert LangChain Documents → our standardized dict format
    documents = []
    for doc in lc_docs:
        text = doc.page_content
        if not text or not text.strip():
            continue

        documents.append({
            "title": doc.metadata.get("title", "Text not found at source"),
            "text": text,
            "source": source,
            "creation_date": doc.metadata.get("creation_date", ""),
            "raw_data": doc.metadata,
        })

    return documents
