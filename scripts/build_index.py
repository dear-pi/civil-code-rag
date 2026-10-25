"""Build a persistent Chroma index for one chunking strategy (used by ccr-ask).

Usage: python scripts/build_index.py [strategy]   # fixed | article | semantic
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from civil_code_rag.chunking import build_chunks  # noqa: E402
from civil_code_rag.corpus import load_corpus  # noqa: E402
from civil_code_rag.embeddings import LocalBgeEmbedder  # noqa: E402
from civil_code_rag.retriever import build_retriever  # noqa: E402

CORPUS = ROOT / "data" / "corpus" / "civil_code_2020.json"


def main() -> None:
    strategy = sys.argv[1] if len(sys.argv) > 1 else "article"
    articles = load_corpus(CORPUS)
    embedder = LocalBgeEmbedder()
    chunks = build_chunks(strategy, articles, embedder)
    retriever = build_retriever("chroma", ROOT / ".index" / strategy, strategy)
    retriever.add(chunks, embedder.encode_documents([c.text for c in chunks]))
    print(f"indexed {len(retriever)} chunks for strategy '{strategy}'")


if __name__ == "__main__":
    main()
