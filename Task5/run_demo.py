from __future__ import annotations
import argparse,json
from datetime import datetime
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parent; cases=json.loads((ROOT/'test_cases.json').read_text(encoding='utf-8'))
def ask(client,url,q,mode):
    r=client.post(url.rstrip('/')+'/ask',json={'question':q,'protection_mode':mode}); r.raise_for_status(); return r.json()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-url',default='http://127.0.0.1:8000'); ap.add_argument('--mode',default='all',choices=['pre_prompt','sanitize','drop','all']); a=ap.parse_args(); sections=['# Task 5 demo log','',f'- Protection mode: `{a.mode}`','']; n=1
    with httpx.Client(timeout=180) as c:
        for label,key in [('Expected useful answer','successful'),('Expected refusal / filtered result','negative_or_filtered')]:
            for q in cases[key]:
                p=ask(c,a.base_url,q,a.mode); sec=p.get('security',{}); print(f"{n}/10: {q} -> {p.get('answer','')[:90]}")
                sections += [f'## {n}. {label}','',f'**Question:** {q}','',f'**Answer:** {p.get("answer","")}','','**Sources:**']
                sections += [f'- `{x["source"]}` (score={x["score"]:.4f})' for x in p.get('sources',[])] or ['- none']
                sections += ['','**Security events:**']
                sections += [f'- `{e["action"]}`: `{e["source"]}`' for e in sec.get('events',[])] or ['- none']
                sections += ['',f'**Output blocked:** `{sec.get("output_blocked",False)}`','']; n+=1
    out=ROOT/'logs'/f'demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{a.mode}.md'; out.write_text('\n'.join(sections),encoding='utf-8'); print('Saved:',out)
if __name__=='__main__': main()
