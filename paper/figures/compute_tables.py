import json, sys, collections, statistics as st
sb = json.load(open(sys.argv[1]))
rbf = json.load(open(sys.argv[2])); rb = rbf["skills"]; rb=list(rb.values()) if isinstance(rb,dict) else rb

print("#"*70,"\n#13 LIFECYCLE EVIDENCE — what drives status (fixer)\n","#"*70)
by = collections.defaultdict(list)
for s in rb: by[s.get("status")].append(s)
for stt in ("active","cold","quarantined"):
    g=by.get(stt,[])
    if not g: continue
    def m(f): 
        v=[ (x.get(f) or 0) for x in g]; return sum(v)/len(v)
    upl=[x.get("estimated_uplift") for x in g if x.get("estimated_uplift") is not None]
    print(f"\n[{stt}]  n={len(g)}")
    print(f"   mean selection_count={m('selection_count'):.2f}  mean paired_n={m('paired_n'):.2f}  mean success_with_skill={m('success_with_skill'):.2f}")
    print(f"   mean naked_baseline_success={m('naked_baseline_success'):.2f}  mean harm_count={m('harm_count'):.2f}")
    print(f"   estimated_uplift: n_with={len(upl)} mean={st.mean(upl):.3f}" if upl else "   no uplift")
    print(f"   paired_n>=4: {sum(1 for x in g if (x.get('paired_n') or 0)>=4)}/{len(g)}")

print("\n--- representative lifecycle rows (top evidenced per status, by paired_n) ---")
for stt in ("active","cold","quarantined"):
    g=sorted(by.get(stt,[]), key=lambda x:-(x.get("paired_n") or 0))[:4]
    for s in g:
        print(f"  [{stt:11}] {s['name'][:34]:34} fc={s.get('failure_class','')[:14]:14} sel={s.get('selection_count'):>3} nk={s.get('naked_baseline_success'):>2} ok={s.get('success_with_skill'):>3} harm={s.get('harm_count'):>2} upl={s.get('estimated_uplift')}")

print("\n"+"#"*70,"\n#18 PROVENANCE & DUPLICATION (generator)\n","#"*70)
# reusable vs single-task
dt=collections.Counter(s.get("distinct_tasks") for s in sb)
print("distinct_tasks distribution:", dict(sorted(dt.items())))
print("single-task skills:", dt.get(1,0), "/", len(sb), "= %.0f%%"%(100*dt.get(1,0)/len(sb)))
print("reusable (>=2 tasks):", sum(v for k,v in dt.items() if k and k>=2))
# provenance: seed-source vs self-play origin per skill (from first example source)
def origin(s):
    exs=s.get("examples") or []
    srcs={e.get("source") for e in exs if isinstance(e,dict)}
    real={"human","human_edited_lm","qwen7b","gpt_oss_20b"}
    gen={"generator_skilled","generator_exploration"}
    if srcs & real and not (srcs & gen): return "seed"
    if srcs & gen and not (srcs & real): return "self-play"
    if srcs & real and srcs & gen: return "mixed"
    return "other"
orig=collections.Counter(origin(s) for s in sb)
print("origin (by example sources):", dict(orig))
# duplication: cluster by identical structural signature
def sig(s):
    stc=s.get("structural") or {}
    return (tuple(sorted(stc.get("site_types") or [])), tuple(sorted(stc.get("operators") or [])), tuple(sorted(stc.get("parent_types") or [])))
clusters=collections.Counter(sig(s) for s in sb)
dup_groups={k:v for k,v in clusters.items() if v>1}
in_dup=sum(v for v in dup_groups.values())
print(f"structural signatures: {len(clusters)} distinct over {len(sb)} skills")
print(f"skills sharing a signature with >=1 other: {in_dup} ({100*in_dup/len(sb):.0f}%) in {len(dup_groups)} groups; largest group={max(clusters.values())}")
# top reusable skills (for the table sample)
print("\n--- top generator skills by distinct_tasks (reusable head) ---")
for s in sorted(sb, key=lambda x:-(x.get("distinct_tasks") or 0))[:8]:
    exs=s.get("examples") or []
    srcs=sorted({e.get("source") for e in exs if isinstance(e,dict)})
    st_types=(s.get("structural") or {}).get("site_types")
    print(f"  {s['name'][:38]:38} tasks={s.get('distinct_tasks'):>2} apps={s.get('valid_applications'):>2} site={st_types} srcs={srcs}")
