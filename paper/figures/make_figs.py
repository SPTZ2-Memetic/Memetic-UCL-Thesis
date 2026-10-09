#!/usr/bin/env python3
"""Thesis Results figures from build_results CSVs. Okabe-Ito palette, no dual axis,
recessive grid, direct labels on bars. Writes PDF (thesis) + PNG (preview)."""
import csv, os, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Input: directory of build_results CSVs. Output: where figures are written.
# Override via MEMETIC_RESULTS_DIR / MEMETIC_FIGS_DIR.
IN  = os.environ.get("MEMETIC_RESULTS_DIR", "runs/results")
OUT = os.environ.get("MEMETIC_FIGS_DIR", "paper/figures/out")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito
BLUE="#0072B2"; ORANGE="#E69F00"; GREEN="#009E73"; SKY="#56B4E9"
VERM="#D55E00"; PURP="#CC79A7"; YEL="#F0E442"; GREY="#999999"; INK="#222222"
ARMC={"naked":BLUE,"synth":ORANGE,"memory":GREEN}

plt.rcParams.update({
    "font.size":11,"font.family":"serif","axes.edgecolor":"#888888",
    "axes.linewidth":0.8,"axes.grid":True,"grid.color":"#dddddd",
    "grid.linewidth":0.6,"figure.dpi":110,"savefig.bbox":"tight",
})
def style(ax, ygrid_only=True):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if ygrid_only: ax.xaxis.grid(False)
def save(fig, name):
    for ext in ("pdf","png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig); print("  ->", name)

def read(f):
    with open(os.path.join(IN,f)) as fh: return list(csv.DictReader(fh))
import math
def ffloat(x):
    try: return float(x)
    except: return math.nan
def roll(xs, w=50):
    out=[]
    for i in range(len(xs)):
        s=max(0,i-w+1); win=[v for v in xs[s:i+1] if not math.isnan(v)]
        out.append(sum(win)/len(win) if win else math.nan)
    return out

L = read("longitudinal.csv")
rnd=[int(r["round"]) for r in L]
sk=[int(r["skillbank_size"]) for r in L]; rb=[int(r["repairbank_size"]) for r in L]
cold=[int(r["cold"]) for r in L]; act=[int(r["active"]) for r in L]; quar=[int(r["quarantined"]) for r in L]
front=[ffloat(r["frontier_pct"]) for r in L]; msr=[ffloat(r["mean_sr"]) for r in L]

# F2 bank growth (two panels, no dual axis)
fig,(a1,a2)=plt.subplots(1,2,figsize=(8,3.2))
a1.plot(rnd,sk,color=BLUE,lw=2); style(a1)
a1.set_title("Generator SkillBank"); a1.set_xlabel("round"); a1.set_ylabel("skills")
a1.annotate(f"{sk[-1]}",(rnd[-1],sk[-1]),xytext=(-4,4),textcoords="offset points",ha="right",color=BLUE)
a1.annotate(f"{sk[0]}",(rnd[0],sk[0]),xytext=(4,-10),textcoords="offset points",color=BLUE)
a2.plot(rnd,rb,color=GREEN,lw=2); style(a2)
a2.set_title("Fixer RepairBank"); a2.set_xlabel("round"); a2.set_ylabel("skills")
a2.annotate(f"{rb[-1]}",(rnd[-1],rb[-1]),xytext=(-4,4),textcoords="offset points",ha="right",color=GREEN)
fig.tight_layout(); save(fig,"fig2_bank_growth")

# F7 fixer lifecycle stacked
fig,ax=plt.subplots(figsize=(6.4,3.4))
ax.stackplot(rnd,cold,act,quar,labels=["cold","active","quarantined"],
             colors=[SKY,GREEN,VERM],edgecolor="white",linewidth=0.2)
style(ax); ax.set_xlabel("round"); ax.set_ylabel("repair skills")
ax.set_title("Fixer repair-skill lifecycle"); ax.legend(loc="upper left",frameon=False,fontsize=9)
fig.tight_layout(); save(fig,"fig7_lifecycle")

# F4 frontier %, F5 mean sr (rolling), two panels
fig,(a1,a2)=plt.subplots(1,2,figsize=(8,3.2))
a1.plot(rnd,front,color="#cccccc",lw=0.6); a1.plot(rnd,roll(front),color=BLUE,lw=2)
style(a1); a1.set_title("Frontier rate (0<sr<1)"); a1.set_xlabel("round"); a1.set_ylabel("% of valid bugs")
a2.plot(rnd,msr,color="#cccccc",lw=0.6); a2.plot(rnd,roll(msr),color=GREEN,lw=2)
style(a2); a2.set_title("Mean round solve-rate"); a2.set_xlabel("round"); a2.set_ylabel("solve-rate")
a2.set_ylim(0,1)
fig.tight_layout(); save(fig,"fig45_frontier_meansr")

# F3 generator site-type stacked area over checkpoints
S=[r for r in read("longitudinal_sites.csv") if r["round"].isdigit()]
srnd=[int(r["round"]) for r in S]
cols=["assignment","call","return","branch","loop_header"]
rest=[c for c in S[0] if c not in cols+["round"]]
series={c:[int(r[c]) for r in S] for c in cols}
other=[sum(int(r[c]) for c in rest) for r in S]
fig,ax=plt.subplots(figsize=(6.6,3.6))
pal=[BLUE,ORANGE,GREEN,PURP,SKY,GREY]
ax.stackplot(srnd,*[series[c] for c in cols],other,labels=cols+["other"],
             colors=pal,edgecolor="white",linewidth=0.3)
style(ax); ax.set_xlabel("round"); ax.set_ylabel("generator skills")
ax.set_title("Generator skill site-type composition")
ax.legend(loc="upper left",frameon=False,fontsize=8,ncol=2)
fig.tight_layout(); save(fig,"fig3_sites")

# F8 generalisation scatter: distinct_tasks vs valid_applications
G=read("gen_skills.csv")
dt=[int(r["distinct_tasks"]) for r in G]; va=[int(r["valid_applications"]) for r in G]
fig,ax=plt.subplots(figsize=(5.4,4.2))
ax.scatter(dt,va,s=22,color=BLUE,alpha=0.5,edgecolor="none")
m=max(max(dt),max(va)); ax.plot([0,m],[0,m],color=GREY,lw=1,ls="--",label="valid apps = distinct tasks")
style(ax); ax.set_xlabel("distinct tasks"); ax.set_ylabel("valid applications")
ax.set_title("Generator skill generalisation"); ax.legend(frameon=False,fontsize=9,loc="lower right")
fig.tight_layout(); save(fig,"fig8_gen_scatter")

# F9 reuse histogram
ruse=[int(r["recent_use_count"]) for r in G]
fig,ax=plt.subplots(figsize=(5.8,3.4))
ax.hist(ruse,bins=30,color=BLUE,edgecolor="white")
style(ax); ax.set_yscale("log"); ax.set_xlabel("recent-use count"); ax.set_ylabel("skills (log)")
ax.set_title("Generator skill reuse distribution")
fig.tight_layout(); save(fig,"fig9_reuse_hist")

# F10 fixer utility/harm: selection vs uplift, color by status, size by harm
R=read("repair_skills.csv")
def fnum(x):
    try: return float(x)
    except: return 0.0
stc={"active":GREEN,"cold":SKY,"quarantined":VERM}
fig,ax=plt.subplots(figsize=(6.2,4.2))
for stt in ("cold","active","quarantined"):
    xs=[int(r["selection_count"]) for r in R if r["status"]==stt]
    ys=[fnum(r["estimated_uplift"]) for r in R if r["status"]==stt]
    hs=[int(r["harm_count"]) for r in R if r["status"]==stt]
    ax.scatter(xs,ys,s=[12+8*h for h in hs],color=stc[stt],alpha=0.55,edgecolor="none",label=stt)
style(ax); ax.set_xlabel("selection count"); ax.set_ylabel("loop uplift proxy")
ax.set_title("Fixer repair-skill utility (size $\\propto$ harm)")
ax.legend(frameon=False,fontsize=9,loc="lower right")
fig.tight_layout(); save(fig,"fig10_utility_harm")

# --- EVAL ---
B=read("eval_by_bucket.csv")
# average the two passes
agg=collections.defaultdict(lambda: collections.defaultdict(list))
for r in B: agg[r["bucket"]][r["arm"]].append(float(r["solved_pct"]))
buckets=["assertion","crash","other"]; arms=["naked","synth","memory"]
# pooled from arms table numbers
pooled={"naked":64.1,"synth":57.6,"memory":59.0}
def gb(data, cats, title, name, pooled_row=None):
    import numpy as np
    x=list(range(len(cats))); w=0.26
    fig,ax=plt.subplots(figsize=(7.2,3.8))
    for i,arm in enumerate(arms):
        vals=[data[c][arm] for c in cats]
        bars=ax.bar([xx+(i-1)*w for xx in x],vals,w,color=ARMC[arm],label=arm,edgecolor="white",linewidth=0.5)
        for b,v in zip(bars,vals):
            ax.annotate(f"{v:.0f}",(b.get_x()+b.get_width()/2,v),xytext=(0,2),
                        textcoords="offset points",ha="center",fontsize=8,color=INK)
    style(ax); ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("solved@k (%)")
    ax.set_ylim(0,85); ax.set_title(title); ax.legend(frameon=False,fontsize=9,ncol=3,loc="upper right")
    fig.tight_layout(); save(fig,name)

# strata figure (+ pooled)
strat={b:{a:sum(agg[b][a])/len(agg[b][a]) for a in arms} for b in buckets}
strat["pooled"]=pooled
gb(strat,buckets+["pooled"],"Held-out solved@k by failure class","fig12_eval_strata")

# by source
SRC=read("eval_by_source.csv")
sdata=collections.defaultdict(dict)
for r in SRC: sdata[r["source"]][r["arm"]]=float(r["solved_pct"])
order=["human_edited_lm","human","gpt_oss_20b","qwen7b"]
gb(sdata,order,"Held-out solved@k by bug source","fig11_eval_by_source")

# escalation compact bar
fig,ax=plt.subplots(figsize=(4.8,3.4))
labs=["naked","escalate\n→synth","escalate\n→memory"]; vals=[64.1,69.9,69.9]
cols=[BLUE,GREEN,GREEN]
bars=ax.bar(labs,vals,color=cols,edgecolor="white",width=0.6)
for b,v in zip(bars,vals):
    ax.annotate(f"{v:.1f}",(b.get_x()+b.get_width()/2,v),xytext=(0,2),textcoords="offset points",ha="center",fontsize=9)
style(ax); ax.set_ylabel("solved@k (%)"); ax.set_ylim(0,80)
ax.set_title("Escalation: naked first, memory on a miss")
fig.tight_layout(); save(fig,"fig12b_escalation")

print("\nAll figures in", OUT)
