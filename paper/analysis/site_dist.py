import json, collections
sb=json.load(open("runs/selfplay_thesis/skillbank.json"))
def site(s):
    st=(s.get("structural") or {}).get("site_types") or []
    return st[0] if st else "none"
c=collections.Counter(site(s) for s in sb)
print("site_type counts (all):")
for k,v in c.most_common(): print(f"  {k}: {v}")
# multi-task by site
multi=collections.Counter(site(s) for s in sb if (s.get("distinct_tasks") or 0)>=2)
print("\nmulti-task (>=2) by site:", dict(multi.most_common()))
