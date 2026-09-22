from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source_docs"
OUTPUT_DIR = ROOT / "knowledge_base"
TERMS_FILE = ROOT / "terms_map.json"


def clean_text(text: str) -> str:
    """Remove basic HTML / reference noise and normalize whitespace."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def replace_terms(text: str, terms: dict[str, str]) -> str:
    """Replace longest terms first to avoid partial replacements."""
    for source, target in sorted(
        terms.items(), key=lambda item: len(item[0]), reverse=True
    ):
        pattern = re.compile(
            rf"(?<!\w){re.escape(source)}(?!\w)",
            flags=re.IGNORECASE,
        )
        text = pattern.sub(target, text)
    return text


def main() -> None:
    terms = json.loads(TERMS_FILE.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_DIR.exists():
        raise SystemExit(
            "Task2/source_docs does not exist. "
            "Put cleaned/paraphrased source .txt/.md documents there first."
        )

    for source_path in sorted(SOURCE_DIR.glob("*")):
        if source_path.suffix.lower() not in {".txt", ".md"}:
            continue

        raw = source_path.read_text(encoding="utf-8")
        transformed = replace_terms(clean_text(raw), terms)

        output_path = OUTPUT_DIR / f"{source_path.stem}.md"
        output_path.write_text(transformed + "\n", encoding="utf-8")
        print(f"{source_path.name} -> {output_path.name}")


if __name__ == "__main__":
    main()
