"""Interactive CLI: retrieve (and, with an API key, answer) a question.

Usage:
    ccr-ask "诉讼时效是几年" [--strategy article] [--top-k 5]
          [--refuse-threshold 2.0] [--retriever chroma|numpy]

Without ``CCR_LLM_API_KEY`` the command runs in retrieval-only mode: it
prints the reranked chunks with scores — still useful and fully offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chunking import build_chunks
from .corpus import load_corpus
from .embeddings import LocalBgeEmbedder
from .generation import Generator
from .rerank import CrossEncoderReranker
from .retriever import build_retriever

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = ROOT / "data" / "corpus" / "civil_code_2020.json"


def main() -> int:
    ap = argparse.ArgumentParser(prog="ccr-ask", description=__doc__)
    ap.add_argument("question")
    ap.add_argument("--strategy", default="article", choices=["fixed", "article", "semantic"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--refuse-threshold", type=float, default=None)
    ap.add_argument("--retriever", default="chroma", choices=["chroma", "numpy"])
    args = ap.parse_args()

    embedder = LocalBgeEmbedder()
    reranker = CrossEncoderReranker()

    if args.retriever == "chroma":
        index_dir = ROOT / ".index" / args.strategy
        retriever = build_retriever("chroma", index_dir, args.strategy)
        if len(retriever) == 0:
            print(f"index for '{args.strategy}' empty — building in place...", file=sys.stderr)
            articles = load_corpus(CORPUS)
            chunks = build_chunks(args.strategy, articles)
            retriever.add(chunks, embedder.encode_documents([c.text for c in chunks]))
        qvec = embedder.encode_query(args.question)
        hits = retriever.search(qvec, 20)
    else:
        articles = load_corpus(CORPUS)
        chunks = build_chunks(args.strategy, articles)
        retriever = build_retriever("numpy")
        retriever.add(chunks, embedder.encode_documents([c.text for c in chunks]))
        qvec = embedder.encode_query(args.question)
        hits = retriever.search(qvec, 20)

    reranked = reranker.rerank(args.question, [c for c, _ in hits], args.top_k)
    if not reranked:
        print("no chunks retrieved")
        return 1

    top_score = reranked[0][1]
    threshold = args.refuse_threshold
    if threshold is None:
        import json

        calib = ROOT / "results" / "calibration.json"
        if calib.exists():
            threshold = json.loads(calib.read_text(encoding="utf-8"))["refusal"][
                "threshold_rerank_logit"
            ]
        else:
            threshold = 0.0

    gen = Generator()
    if not gen.available:
        print(f"[retrieval-only mode: set CCR_LLM_API_KEY for generation]  τ={threshold:.2f}")
        for c, s in reranked:
            print(f"  {s:7.3f}  {c.citation:<14} {c.text[:60]}")
        return 0

    answer = gen.answer(args.question, [c for c, _ in reranked], top_score, threshold)
    if answer.refused:
        print(f"[拒答] {answer.refusal_reason}")
        print("最相关条文（仅供参考）：")
        for c, s in reranked[:3]:
            print(f"  {s:7.3f}  {c.citation}  {c.text[:60]}")
        return 0
    print(answer.text)
    if answer.unfaithful_citations:
        print(f"[警告] 引用了未提供的条文: {answer.unfaithful_citations}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
