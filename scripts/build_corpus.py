"""Extract Civil Code text from the official print-PDF and parse it.

Usage: python scripts/build_corpus.py [pdf_path] [out_json]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from civil_code_rag.corpus import assert_sequence, parse, save_corpus  # noqa: E402


def extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        t = page.extract_text() or ""
        lines = t.splitlines()
        # drop the two running header lines every page carries
        if lines and "_中国人大网" not in lines[0] and lines[0].strip():
            pages.append("\n".join(lines))
        else:
            pages.append("\n".join(lines[2:] if len(lines) > 2 else []))
    return "\n".join(pages)


def main() -> None:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "civil_code_fao.pdf"
    out = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else ROOT / "data" / "corpus" / "civil_code_2020.json"
    )

    text = extract_pdf_text(pdf)
    raw_txt = out.with_suffix(".txt")
    raw_txt.parent.mkdir(parents=True, exist_ok=True)
    raw_txt.write_text(text, encoding="utf-8")

    articles = parse(text)
    assert_sequence(articles)
    save_corpus(articles, out)

    total_chars = sum(len(a.text) for a in articles)
    parts = sorted({a.part for a in articles})
    print(f"parsed {len(articles)} articles, {total_chars} chars")
    print("parts:", " / ".join(parts))
    print("first:", articles[0].label, articles[0].text[:40])
    print("last :", articles[-1].label, articles[-1].text[:40])


if __name__ == "__main__":
    main()
