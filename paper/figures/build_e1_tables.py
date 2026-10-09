import json, sys, collections
sb = json.load(open(sys.argv[1]))
rbf = json.load(open(sys.argv[2])); rb=rbf["skills"]; rb=list(rb.values()) if isinstance(rb,dict) else rb
OUT=sys.argv[3]

def esc(s):
    s=str(s)
    for a,b in [("\\","\\textbackslash "),("_","\\_"),("%","\\%"),("&","\\&"),("#","\\#"),("$","\\$")]:
        s=s.replace(a,b)
    return s
def trunc(s,n): 
    s=str(s); return s if len(s)<=n else s[:n-1]+"\u2026"

# ---------------- #13 lifecycle evidence ----------------
by=collections.defaultdict(list)
for s in rb: by[s.get("status")].append(s)
def rate(s):
    p=s.get("selection_count") or 0
    return (s.get("success_with_skill") or 0)/p if p else None
rows=[]
for s in sorted(by["active"], key=lambda x:-(x.get("selection_count") or 0))[:4]:
    rows.append(s)
# cold: 2 untested (lowest selection, but nonzero retrieval so they were seen), 1 graded-middling
cold_mid=[s for s in by["cold"] if (s.get("selection_count") or 0)>=4]
cold_unt=sorted([s for s in by["cold"] if (s.get("selection_count") or 0)<=1 and (s.get("retrieval_count") or 0)>0], key=lambda x:-(x.get("retrieval_count") or 0))[:2]
rows+=cold_unt
if cold_mid: rows.append(sorted(cold_mid,key=lambda x:-(x.get("harm_count") or 0))[0])
for s in by["quarantined"]: rows.append(s)

with open(f"{OUT}/tab_lifecycle_evidence.tex","w") as f:
    f.write("\\begin{table}[tbp]\n\\centering\n\\small\n")
    f.write("\\begin{tabular}{@{}llrrrrl@{}}\n\\toprule\n")
    f.write("\\textbf{Skill} & \\textbf{Failure class} & \\textbf{Sel.} & \\textbf{Succ.} & \\textbf{Harm} & \\textbf{Succ.\\ rate} & \\textbf{Status} \\\\\n\\midrule\n")
    last=None
    for s in rows:
        if s.get("status")!=last and last is not None: f.write("\\addlinespace\n")
        last=s.get("status")
        r=rate(s); rs="%.2f"%r if r is not None else "--"
        f.write(" & ".join([esc(trunc(s.get("name"),30)), esc(s.get("failure_class","")),
                 str(s.get("selection_count") or 0), str(s.get("success_with_skill") or 0),
                 str(s.get("harm_count") or 0), rs, esc(s.get("status"))])+" \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
    f.write("\\caption{Representative repair skills and the evidence behind their lifecycle status. "
            "\\emph{Sel.}\\ is the number of paired trials (selections); \\emph{Succ.rate} is successes over selections. "
            "A skill needs at least four trials to be graded: of $1{,}298$ skills, $294$ reach \\texttt{active} "
            "(success rate $\\geq 0.34$), $2$ are \\texttt{quarantined} (rate $<0.15$ with net harm), and $1{,}002$ "
            "stay \\texttt{cold}---overwhelmingly because they never accumulate four trials, not because they fail the gate.}\n")
    f.write("\\label{tab:lifecycle-evidence}\n\\end{table}\n")

# ---------------- #18 generator provenance + duplication ----------------
def origin(s):
    srcs={e.get("source") for e in (s.get("examples") or []) if isinstance(e,dict)}
    real={"human","human_edited_lm","qwen7b","gpt_oss_20b"}; gen={"generator_skilled","generator_exploration"}
    if srcs&real and not srcs&gen: return "seed"
    if srcs&gen and not srcs&real: return "self-play"
    if srcs&real and srcs&gen: return "mixed"
    return "?"
orig=collections.Counter(origin(s) for s in sb)
dt=collections.Counter(s.get("distinct_tasks") for s in sb)
single=dt.get(1,0); reuse=sum(v for k,v in dt.items() if k and k>=2)
def srcset(s):
    m={"human":"H","human_edited_lm":"H-lm","qwen7b":"Qw","gpt_oss_20b":"OSS","generator_skilled":"sp","generator_exploration":"sp-x"}
    ss=[]
    for e in (s.get("examples") or []):
        if isinstance(e,dict): ss.append(m.get(e.get("source"),"?"))
    seen=[]; 
    for x in ss:
        if x not in seen: seen.append(x)
    return ",".join(seen)
head=sorted(sb, key=lambda x:-(x.get("distinct_tasks") or 0))[:8]
with open(f"{OUT}/tab_provenance_gen.tex","w") as f:
    f.write("\\begin{table}[tbp]\n\\centering\n\\small\n")
    f.write("\\begin{tabular}{@{}llrl@{}}\n\\toprule\n")
    f.write("\\textbf{Skill} & \\textbf{Site} & \\textbf{Tasks} & \\textbf{Sources} \\\\\n\\midrule\n")
    for s in head:
        site=",".join((s.get("structural") or {}).get("site_types") or []) or "--"
        f.write(" & ".join([esc(trunc(s.get("name"),34)), esc(site), str(s.get("distinct_tasks") or 0), esc(srcset(s))])+" \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
    f.write("\\caption{The most broadly-reused generator skills (by distinct tasks). "
            "Source codes: H human, H-lm human-edited-LM, Qw Qwen-7B, OSS GPT-OSS-20B, sp self-play. "
            "The reusable head is dominated by \\texttt{return}-site mechanisms, several of them variations on an incorrect return value.}\n")
    f.write("\\label{tab:provenance-gen}\n\\end{table}\n")

print("ORIGIN:", dict(orig))
print("single-task:", single, "/", len(sb), "=%.0f%%"%(100*single/len(sb)), "| reusable>=2:", reuse, "| max tasks:", max(dt))
print("wrote tab_lifecycle_evidence.tex, tab_provenance_gen.tex")
