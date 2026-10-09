import json, collections
sb=json.load(open("runs/selfplay_thesis/skillbank.json"))
ru=[s.get("recent_use_count") or 0 for s in sb]
dt=[s.get("distinct_tasks") or 0 for s in sb]
# reuse bins
bins=[(0,0),(1,1),(2,4),(5,9),(10,19),(20,49),(50,999)]
print("RECENT_USE_COUNT (how many times a skill was retrieved to guide generation):")
for lo,hi in bins:
    c=sum(1 for x in ru if lo<=x<=hi)
    print(f"  {lo}-{hi if hi<999 else '+'}: {c} skills")
print(f"  total reuse sum={sum(ru)}; top-10 skills hold {sum(sorted(ru,reverse=True)[:10])} ({100*sum(sorted(ru,reverse=True)[:10])/sum(ru):.0f}%)")
print("\nDISTINCT_TASKS (how many different tasks the skill was applied to):")
d=collections.Counter(dt)
for k in sorted(d): print(f"  {k} task(s): {d[k]} skills")
