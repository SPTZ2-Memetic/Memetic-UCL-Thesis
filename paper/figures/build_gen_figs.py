import json, collections, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.color":"#E6E6E6","grid.linewidth":0.8,"axes.axisbelow":True,
    "figure.dpi":140,"savefig.bbox":"tight","font.family":"DejaVu Sans"})

OK = {"return":"#0072B2","call":"#E69F00","assignment":"#009E73","branch":"#CC79A7","other":"#9A9A9A"}
ORDER = ["assignment","call","return","branch","other"]
def site(s):
    st=(s.get("structural") or {}).get("site_types") or []
    t=st[0] if st else "none"
    return t if t in OK else "other"

sb=json.load(open("e1data/runs/selfplay_thesis/skillbank.json"))
dt=np.array([s.get("distinct_tasks") or 0 for s in sb])
va=np.array([s.get("valid_applications") or 0 for s in sb])
ru=np.array([s.get("recent_use_count") or 0 for s in sb])
st=[site(s) for s in sb]

# ============ FIGURE 1: generalisation (scatter + stacked bar) ============
fig,(axA,axB)=plt.subplots(1,2,figsize=(10,4.3))

# -- (a) scatter: distinct tasks vs valid applications, size=reuse, colour=site
rng=np.random.default_rng(0)
for cat in ORDER:
    idx=[i for i in range(len(sb)) if st[i]==cat]
    if not idx: continue
    jx=dt[idx]+rng.uniform(-0.18,0.18,len(idx))
    jy=va[idx]+rng.uniform(-0.18,0.18,len(idx))
    sz=20+ru[idx]*3.0
    axA.scatter(jx,jy,s=sz,c=OK[cat],alpha=0.55,edgecolors="white",linewidths=0.4,label=cat,zorder=3)
lim=max(dt.max(),va.max())+1
axA.plot([0.5,lim],[0.5,lim],color="#BBBBBB",lw=1.0,ls="--",zorder=1)
axA.text(lim*0.72,lim*0.80,"identity line\n(apps = tasks)",color="#888888",fontsize=8,ha="left",va="center")
axA.set_xlabel("Distinct tasks the skill was applied to")
axA.set_ylabel("Valid bug applications")
axA.set_title("(a) Tasks spanned vs applications\nmarker size ∝ times retrieved",fontsize=10,loc="left")
axA.set_xlim(0,lim); axA.set_ylim(0,lim)

# -- (b) stacked bar: distinct-task buckets by site
buckets=[("1",lambda x:x==1),("2",lambda x:x==2),("3–4",lambda x:3<=x<=4),
         ("5–9",lambda x:5<=x<=9),("10+",lambda x:x>=10)]
bl=[b[0] for b in buckets]; xpos=np.arange(len(buckets))
bottom=np.zeros(len(buckets))
for cat in ORDER:
    vals=[]
    for _,fn in buckets:
        vals.append(sum(1 for i in range(len(sb)) if st[i]==cat and fn(dt[i])))
    axB.bar(xpos,vals,bottom=bottom,color=OK[cat],width=0.74,label=cat,edgecolor="white",linewidth=0.6)
    bottom+=np.array(vals)
axB.set_xticks(xpos); axB.set_xticklabels(bl)
axB.set_xlabel("Distinct tasks a skill is applied to")
axB.set_ylabel("Number of skills")
axB.set_title("(b) Most skills target one task;\na reusable tail spans many",fontsize=10,loc="left")
# annotate the big bar
axB.text(0,bottom[0]+6,f"{int(bottom[0])}",ha="center",fontsize=9,color="#333333")
axB.set_ylim(0,bottom[0]*1.12)
axB.grid(axis="x",visible=False)

handles=[Patch(facecolor=OK[c],label=c) for c in ORDER]
fig.legend(handles=handles,title="construct (site) type",loc="lower center",ncol=5,
           frameon=False,bbox_to_anchor=(0.5,-0.04),fontsize=9,title_fontsize=9)
fig.tight_layout(rect=[0,0.04,1,1])
fig.savefig("fig_gen_generalise.pdf"); fig.savefig("fig_gen_generalise.png",dpi=150)
print("wrote fig_gen_generalise")

# ============ FIGURE 2: reuse histogram ============
fig2,ax=plt.subplots(figsize=(6.2,4.0))
rb=[("0",lambda x:x==0),("1",lambda x:x==1),("2–4",lambda x:2<=x<=4),
    ("5–9",lambda x:5<=x<=9),("10–19",lambda x:10<=x<=19),("20–49",lambda x:20<=x<=49),
    ("50+",lambda x:x>=50)]
vals=[sum(1 for v in ru if fn(v)) for _,fn in rb]
xp=np.arange(len(rb))
bars=ax.bar(xp,vals,color="#0072B2",width=0.72,edgecolor="white",linewidth=0.6)
for i,v in enumerate(vals):
    ax.text(xp[i],v+3,str(v),ha="center",fontsize=8.5,color="#333333")
ax.set_xticks(xp); ax.set_xticklabels([r[0] for r in rb])
ax.set_xlabel("Times the skill was retrieved to guide a bug  (reuse count)")
ax.set_ylabel("Number of skills")
ax.set_title("Skill reuse is concentrated in a small head",fontsize=11,loc="left")
ax.grid(axis="x",visible=False)
top10=sum(sorted(ru,reverse=True)[:10]); tot=ru.sum()
ax.text(0.97,0.90,f"Top 10 skills account for\n{100*top10/tot:.0f}% of all retrievals",
        transform=ax.transAxes,ha="right",va="top",fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4",fc="#F2F2F2",ec="#CCCCCC"))
ax.set_ylim(0,max(vals)*1.15)
fig2.tight_layout()
fig2.savefig("fig_gen_reuse.pdf"); fig2.savefig("fig_gen_reuse.png",dpi=150)
print("wrote fig_gen_reuse")
