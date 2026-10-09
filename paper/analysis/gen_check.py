import json, sys, collections
sb = json.load(open(sys.argv[1]))
def origin(s):
    srcs={e.get("source") for e in (s.get("examples") or []) if isinstance(e,dict)}
    real={"human","human_edited_lm","qwen7b","gpt_oss_20b"}; gen={"generator_skilled","generator_exploration"}
    if srcs&real and not srcs&gen: return "seed"
    if srcs&gen and not srcs&real: return "selfplay"
    if srcs&real and srcs&gen: return "mixed"
    return "?"
groups=collections.defaultdict(list)
for s in sb: groups[origin(s)].append(s)
print("counts:", {k:len(v) for k,v in groups.items()}, "| total", len(sb))
print("seed+mixed =", len(groups['seed'])+len(groups['mixed']), "(should be ~196 warm-start)\n")

def summ(label, L):
    if not L: return
    dt=[s.get("distinct_tasks") or 0 for s in L]
    single=sum(1 for x in dt if x==1); multi=sum(1 for x in dt if x>=2)
    va=[s.get("valid_applications") or 0 for s in L]
    ru=[s.get("recent_use_count") or 0 for s in L]
    va_multi=sum((s.get("valid_applications") or 0) for s in L if (s.get("distinct_tasks") or 0)>=2)
    ru_multi=sum((s.get("recent_use_count") or 0) for s in L if (s.get("distinct_tasks") or 0)>=2)
    print(f"[{label}] n={len(L)}  single-task={single} ({100*single/len(L):.0f}%)  multi-task={multi} ({100*multi/len(L):.0f}%)  max_tasks={max(dt)}")
    print(f"       valid_apps: total={sum(va)}  from multi-task={va_multi} ({100*va_multi/max(sum(va),1):.0f}%)")
    print(f"       recent_use: total={sum(ru)}  from multi-task={ru_multi} ({100*ru_multi/max(sum(ru),1):.0f}%)")

summ("ALL", sb)
summ("SEED-only (warm-start, 1 example each)", groups['seed'])
summ("SELF-PLAY discovered", groups['selfplay'])
summ("MIXED (seed + self-play)", groups['mixed'])
summ("GROWN = self-play + mixed", groups['selfplay']+groups['mixed'])
# reused-only view
reused=[s for s in sb if (s.get("recent_use_count") or 0)>0]
summ("ACTUALLY REUSED (recent_use_count>0)", reused)
