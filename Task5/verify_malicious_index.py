from __future__ import annotations
import json
from pathlib import Path
p=Path("Task3/index/chunks.jsonl")
if not p.exists(): raise SystemExit("Run: python Task3/build_index.py --force")
chunks=[json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
m=[c for c in chunks if Path(c["source"]).name=="malicious_injection.md"]
if not m: raise SystemExit("FAIL: malicious_injection.md is not indexed")
print(f"PASS: malicious_injection.md is indexed as {len(m)} chunk(s)")
for c in m: print('-',c['chunk_id'],c['source'])
