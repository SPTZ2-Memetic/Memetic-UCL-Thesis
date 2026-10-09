import json, collections
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
passes=[load("runs/eval_thesis/pass1.txt.jsonl"), load("runs/eval_thesis/pass2.txt.jsonl")]
buckets=["assertion","crash","other"]
def solved(rec,arm): 
    v=rec.get(arm); 
    return bool(v and v.get("solved"))
# per bucket, averaged over passes: naked/synth/memory solved@k, and paired rescue/harm/net vs naked
agg=collections.defaultdict(lambda: collections.defaultdict(float))
for P in passes:
    by=collections.defaultdict(list)
    for r in P: by[r["bucket"]].append(r)
    for b in buckets+["POOLED"]:
        rows = P if b=="POOLED" else by[b]
        n=len(rows)
        for arm in ("naked","dx_synth","dx_mem_new"):
            agg[b][arm+"_solved"]+= sum(solved(r,arm) for r in rows)/n*100/2
        for arm,lbl in (("dx_synth","synth"),("dx_mem_new","memory")):
            resc=sum(1 for r in rows if solved(r,arm) and not solved(r,"naked"))
            harm=sum(1 for r in rows if solved(r,"naked") and not solved(r,arm))
            agg[b][lbl+"_rescue"]+=resc/2; agg[b][lbl+"_harm"]+=harm/2; agg[b][lbl+"_net"]+=(resc-harm)/2
        agg[b]["n"]=n
print(f"{'bucket':10} {'n':>4} | {'naked':>6} {'synth':>6} {'mem':>6} | mem: {'resc':>5} {'harm':>5} {'net':>6} | synth net")
for b in buckets+["POOLED"]:
    a=agg[b]
    print(f"{b:10} {int(a['n']):>4} | {a['naked_solved']:5.1f}% {a['dx_synth_solved']:5.1f}% {a['dx_mem_new_solved']:5.1f}% | "
          f"      {a['memory_rescue']:5.1f} {a['memory_harm']:5.1f} {a['memory_net']:6.1f} | {a['synth_net']:6.1f}")
