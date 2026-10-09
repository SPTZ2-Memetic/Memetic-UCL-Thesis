import json,collections
rbf=json.load(open("runs/selfplay_thesis/repairbank.json")); rb=list(rbf["skills"].values()) if isinstance(rbf["skills"],dict) else rbf["skills"]
by=collections.defaultdict(list)
for s in rb: by[s["status"]].append(s)
rows=[]
rows+=sorted(by["active"],key=lambda x:-(x.get("selection_count") or 0))[:4]
rows+=sorted([s for s in by["cold"] if (s.get("selection_count") or 0)<=1 and (s.get("retrieval_count") or 0)>0],key=lambda x:-(x.get("retrieval_count") or 0))[:2]
rows+=[sorted([s for s in by["cold"] if (s.get("selection_count") or 0)>=4],key=lambda x:-(x.get("harm_count") or 0))[0]]
rows+=by["quarantined"]
for s in rows:
    p=s.get("selection_count") or 0; r=(s.get("success_with_skill") or 0)/p if p else None
    print(f"{s['status']:11} | {s['name']!r:42} | {s.get('failure_class'):16} sel={s.get('selection_count')} ok={s.get('success_with_skill')} harm={s.get('harm_count')} rate={('%.2f'%r) if r is not None else '--'}")
