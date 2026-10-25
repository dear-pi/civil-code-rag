"""Validate the golden QA set against the parsed corpus. Exits non-zero on any error."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from civil_code_rag.corpus import load_corpus  # noqa: E402
from civil_code_rag.qa import load_qa, validate_qa  # noqa: E402


def main() -> None:
    corpus = load_corpus(ROOT / "data" / "corpus" / "civil_code_2020.json")
    items = load_qa(ROOT / "data" / "golden" / "qa_set.json")
    errors = validate_qa(items, corpus)
    n_in = sum(1 for i in items if i.scope == "in")
    n_out = len(items) - n_in
    topics = sorted({i.topic for i in items if i.scope == "in"})
    print(f"{len(items)} questions ({n_in} in-scope, {n_out} out-of-scope), {len(topics)} topics")
    if errors:
        print(f"FAILED with {len(errors)} error(s):")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print("all golden articles exist; all answer keywords verified in article text")


if __name__ == "__main__":
    main()
