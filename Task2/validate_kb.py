from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KB_DIR = ROOT / "knowledge_base"
TERMS_FILE = ROOT / "terms_map.json"


def main() -> None:
    documents = sorted(KB_DIR.glob("*.md"))
    if len(documents) < 30:
        raise SystemExit(f"Expected at least 30 documents, found {len(documents)}")

    terms = json.loads(TERMS_FILE.read_text(encoding="utf-8"))

    violations: list[str] = []
    for path in documents:
        text = path.read_text(encoding="utf-8").lower()
        for original in terms:
            # Check distinctive mapped terms. Very short common tokens are skipped.
            if len(original) >= 5 and original.lower() in text:
                violations.append(f"{path.name}: {original}")

    if violations:
        print("Original terms still found:")
        for violation in violations:
            print(f"  - {violation}")
        raise SystemExit(1)

    print(f"OK: {len(documents)} documents")
    print("OK: no mapped source terms found in knowledge_base")


if __name__ == "__main__":
    main()
