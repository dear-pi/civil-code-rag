"""Retrieval evaluation grid: 3 chunking strategies x rerank on/off.

Runs entirely without an LLM API key — retrieval metrics (hit@k, MRR@k,
recall@k) depend only on embeddings + reranker, both local. Two calibration
steps are corpus-driven (never tuned on the QA answers):

- semantic-chunking similarity threshold: midpoint between the median
  adjacent-sentence similarity within articles and across article borders;
- refusal threshold: 5th percentile of in-scope top-1 rerank scores; the
  out-of-scope false-accept rate at that threshold is then reported.

Outputs: results/metrics.csv, results/report.md, results/calibration.json,
results/failures.md.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from civil_code_rag.chunking import build_chunks  # noqa: E402
from civil_code_rag.corpus import load_corpus  # noqa: E402
from civil_code_rag.embeddings import LocalBgeEmbedder  # noqa: E402
from civil_code_rag.metrics import evaluate_runs, rank_of_first_hit  # noqa: E402
from civil_code_rag.qa import load_qa  # noqa: E402
from civil_code_rag.rerank import CrossEncoderReranker  # noqa: E402
from civil_code_rag.retriever import NumpyRetriever  # noqa: E402

CORPUS = ROOT / "data" / "corpus" / "civil_code_2020.json"
QA = ROOT / "data" / "golden" / "qa_set.json"
RESULTS = ROOT / "results"
K_FINAL = 5
K_POOL = 20
STRATEGIES = ("fixed", "article", "semantic")


def calibrate_semantic_threshold(articles, embedder) -> tuple[float, dict]:
    """Corpus-statistical threshold: midpoint between within-article and
    cross-article adjacent-sentence similarity medians."""
    import re

    sents: list[tuple[int, str]] = []
    for a in articles:
        for s in re.split(r"(?<=[。！？；])", a.text):
            if s:
                sents.append((a.number, s))
    sample = sents[: 4000]
    vecs = embedder.encode_documents([s for _, s in sample])
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    same = [
        float(s)
        for (n1, _), (n2, _), s in zip(sample[:-1], sample[1:], sims, strict=True)
        if n1 == n2
    ]
    diff = [
        float(s)
        for (n1, _), (n2, _), s in zip(sample[:-1], sample[1:], sims, strict=True)
        if n1 != n2
    ]
    thr = (statistics.median(same) + statistics.median(diff)) / 2
    stats = {
        "threshold": round(float(thr), 4),
        "within_article_median": round(statistics.median(same), 4),
        "cross_article_median": round(statistics.median(diff), 4),
        "n_pairs": len(same) + len(diff),
    }
    return thr, stats


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    articles = load_corpus(CORPUS)
    qa = [q for q in load_qa(QA) if q.scope == "in"]
    oos = [q for q in load_qa(QA) if q.scope == "out"]

    print("loading embedder (downloads model on first run)...")
    embedder = LocalBgeEmbedder()

    sem_thr, sem_stats = calibrate_semantic_threshold(articles, embedder)
    print(f"semantic threshold calibrated on corpus stats: {sem_stats}")

    reranker = CrossEncoderReranker()
    print("reranker ready check (downloads model on first run)...")
    reranker._load()  # noqa: SLF001 — warm-up + fail fast before the long loop

    query_vecs = {q.id: embedder.encode_query(q.question) for q in qa + oos}

    rows = []
    per_q: dict[tuple[str, bool], dict[str, dict]] = {}
    oos_scores: dict[str, list[float]] = {"cosine": [], "rerank": []}
    in_top1_rerank: list[float] = []
    chunk_stats: dict[str, dict] = {}

    for strategy in STRATEGIES:
        t0 = time.time()
        if strategy == "semantic":
            chunks = build_chunks(
                "semantic", articles, lambda t: embedder.encode_documents(t), threshold=sem_thr
            )
        else:
            chunks = build_chunks(strategy, articles)
        lengths = [len(c.text) for c in chunks]
        chunk_stats[strategy] = {
            "chunks": len(chunks),
            "mean_len": round(statistics.mean(lengths), 1),
            "median_len": round(statistics.median(lengths), 1),
            "p95_len": round(float(np.percentile(lengths, 95)), 1),
        }
        print(f"[{strategy}] {len(chunks)} chunks (mean {chunk_stats[strategy]['mean_len']} chars)")

        retriever = NumpyRetriever()
        retriever.add(chunks, embedder.encode_documents([c.text for c in chunks]))

        runs_pool, runs_top5, runs_rerank = [], [], []
        per_q[(strategy, False)] = {}
        per_q[(strategy, True)] = {}
        for q in qa:
            hits = retriever.search(query_vecs[q.id], K_POOL)
            art_sets = [c.article_numbers for c, _ in hits]
            runs_pool.append((art_sets, set(q.golden_articles)))
            runs_top5.append((art_sets[:K_FINAL], set(q.golden_articles)))
            per_q[(strategy, False)][q.id] = {
                "rank": rank_of_first_hit(art_sets[:K_FINAL], set(q.golden_articles)),
                "top1_cos": hits[0][1] if hits else None,
            }
            rer = reranker.rerank(q.question, [c for c, _ in hits], K_FINAL)
            rer_sets = [c.article_numbers for c, _ in rer]
            runs_rerank.append((rer_sets, set(q.golden_articles)))
            per_q[(strategy, True)][q.id] = {
                "rank": rank_of_first_hit(rer_sets, set(q.golden_articles)),
                "top1_rerank": rer[0][1] if rer else None,
            }
            if rer:
                in_top1_rerank.append(rer[0][1])

        for rerank_on, runs in ((False, runs_top5), (True, runs_rerank)):
            m = evaluate_runs(runs, K_FINAL)
            rows.append(
                {
                    "strategy": strategy,
                    "rerank": rerank_on,
                    **{kk: round(v, 4) for kk, v in m.items()},
                }
            )
        pool = evaluate_runs(runs_pool, K_POOL)
        rows.append(
            {
                "strategy": strategy,
                "rerank": False,
                **{kk: round(v, 4) for kk, v in pool.items()},
            }
        )

        for q in oos:
            hits = retriever.search(query_vecs[q.id], K_POOL)
            if hits:
                oos_scores["cosine"].append(hits[0][1])
                rer = reranker.rerank(q.question, [c for c, _ in hits], 1)
                if rer:
                    oos_scores["rerank"].append(rer[0][1])
        print(f"[{strategy}] evaluated in {time.time() - t0:.0f}s")

    # refusal threshold from in-scope top-1 rerank scores (5th percentile)
    tau = float(np.percentile(in_top1_rerank, 5))
    false_accept = sum(1 for s in oos_scores["rerank"] if s >= tau) / len(oos_scores["rerank"])
    calibration = {
        "semantic_threshold": sem_stats,
        "refusal": {
            "threshold_rerank_score": round(tau, 4),
            "in_scope_top1_median": round(float(np.median(in_top1_rerank)), 4),
            "oos_top1_median": round(float(np.median(oos_scores["rerank"])), 4),
            "oos_false_accept_at_threshold": round(false_accept, 4),
            "note": (
                "scores are cross-encoder probabilities (sentence-transformers applies "
                "sigmoid for single-label rerankers), not raw logits; "
                "threshold = 5th percentile of in-scope top-1"
            ),
        },
    }

    (RESULTS / "metrics.csv").write_text(
        "strategy,rerank,hit@k,mrr@k,recall@k,k,n\n"
        + "\n".join(
            f"{r['strategy']},{r['rerank']},{r['hit@k']},{r['mrr@k']},{r['recall@k']},{int(r['k'])},{int(r['n'])}"
            for r in rows
        )
        + "\n",
        encoding="utf-8",
    )
    (RESULTS / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # failure analysis on the best and worst config
    ranked_rows = sorted(
        [r for r in rows if int(r["k"]) == K_FINAL],
        key=lambda r: r["hit@k"],
    )
    worst = ranked_rows[0]
    best = ranked_rows[-1]
    lines = ["# 失败案例（rank > 5 或未命中）", ""]
    pairs = (
        ("最优配置", best["strategy"], best["rerank"]),
        ("最差配置", worst["strategy"], worst["rerank"]),
    )
    for name, strat, rr in pairs:
        lines.append(f"## {name}: {strat} + rerank={rr}")
        misses = 0
        for q in qa:
            info = per_q[(strat, rr)][q.id]
            if info["rank"] is None or info["rank"] > K_FINAL:
                misses += 1
                lines.append(f"- {q.id} [{q.topic}] rank={info['rank']} :: {q.question}")
        if misses == 0:
            lines.append("- 无")
        lines.append("")
    (RESULTS / "failures.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(rows, ensure_ascii=False, indent=1))
    print(json.dumps(calibration["refusal"], ensure_ascii=True))


if __name__ == "__main__":
    main()
