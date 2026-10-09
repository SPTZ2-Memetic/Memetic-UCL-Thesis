import json, sys, collections
sb = json.load(open(sys.argv[1]))
def taskids(s):
    ids=[]
    for e in (s.get("examples") or []):
        if isinstance(e,dict):
            ids.append(e.get("task_id"))
    return ids
print("For the top skills by distinct_tasks — compare the fields:\n")
print(f"{'name':36} {'dtasks':>6} {'vapps':>5} {'reuse':>5} {'#ex':>4} {'uniq_task_ids_in_examples':>24}")
for s in sorted(sb, key=lambda x:-(x.get('distinct_tasks') or 0))[:10]:
    ids=taskids(s); uniq=len(set([i for i in ids if i]))
    print(f"{s['name'][:36]:36} {s.get('distinct_tasks'):>6} {s.get('valid_applications'):>5} {s.get('recent_use_count'):>5} {len(ids):>4} {uniq:>24}")
print()
# relationship across whole bank
import statistics as st
eqA=sum(1 for s in sb if (s.get('distinct_tasks') or 0)==len(set(i for i in taskids(s) if i)))
eqB=sum(1 for s in sb if (s.get('distinct_tasks') or 0)==(s.get('valid_applications') or 0))
print(f"distinct_tasks == #unique task_ids in examples: {eqA}/{len(sb)}")
print(f"distinct_tasks == valid_applications: {eqB}/{len(sb)}")
# does reuse (recent_use_count) exceed valid_applications? (retrieval vs valid)
gt=sum(1 for s in sb if (s.get('recent_use_count') or 0)>(s.get('valid_applications') or 0))
print(f"recent_use_count > valid_applications: {gt}/{len(sb)}  (reuse counts retrievals, not just valid apps)")
# one concrete example: show the example task_ids + sources for the #1 skill
top=sorted(sb, key=lambda x:-(x.get('distinct_tasks') or 0))[0]
print(f"\nTOP skill '{top['name']}': distinct_tasks={top.get('distinct_tasks')} valid_apps={top.get('valid_applications')} reuse={top.get('recent_use_count')} #examples={len(top.get('examples') or [])}")
for e in (top.get('examples') or [])[:6]:
    print("   ex:", {k:e.get(k) for k in ('source','task_id')})
