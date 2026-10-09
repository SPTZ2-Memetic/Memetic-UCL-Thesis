import json, sys, os

def top(path):
    d = json.load(open(path))
    print("TOP-LEVEL type:", type(d).__name__)
    if isinstance(d, dict):
        print("TOP keys:", list(d.keys())[:20])
    return d

def entries(d):
    # find the list/dict of skill entries
    if isinstance(d, list): return d
    for k in ("skills","entries","bank","items","repairs"):
        if isinstance(d, dict) and k in d and isinstance(d[k],(list,dict)):
            v = d[k]
            return list(v.values()) if isinstance(v,dict) else v
    # dict of id->entry?
    if isinstance(d, dict):
        vals=list(d.values())
        if vals and isinstance(vals[0],dict): return vals
    return []

for label,path in [("SKILLBANK (generator)", sys.argv[1]), ("REPAIRBANK (fixer)", sys.argv[2])]:
    print("\n"+"="*70+"\n"+label+"\n"+"="*70)
    d = top(path)
    es = entries(d)
    print("num entries:", len(es))
    if es:
        e = es[0]
        print("sample entry keys:", list(e.keys()))
        print("--- sample entry (truncated values) ---")
        for k,v in e.items():
            s = json.dumps(v, default=str)
            print(f"  {k}: {s[:160]}")
