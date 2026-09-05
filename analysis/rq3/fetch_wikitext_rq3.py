#!/usr/bin/env python3
"""Fetch the fixed WikiText-2 test excerpt used by RQ3 fluency guards."""

from pathlib import Path

from datasets import load_dataset

PROJ = Path(__file__).resolve().parents[2]
OUT = PROJ / "data" / "wikitext_rq3_guard.txt"


def main() -> None:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    lines = [str(row["text"]).strip() for row in dataset if str(row["text"]).strip()]
    # More than enough for 1,001 tokens under every tokenizer in Tier B.
    text = "\n\n".join(lines)
    if len(text) < 20_000:
        raise RuntimeError("WikiText download is unexpectedly short")
    OUT.write_text(text[:100_000] + "\n", encoding="utf-8")
    print(f"saved {OUT} chars={OUT.stat().st_size}")


if __name__ == "__main__":
    main()
