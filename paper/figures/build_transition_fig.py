import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
 "axes.grid":True,"grid.color":"#E6E6E6","grid.linewidth":0.8,"axes.axisbelow":True,
 "figure.dpi":140,"savefig.bbox":"tight","font.family":"DejaVu Sans"})
rounds=[150,300,450,600,750,900,1050,1200,1350]
entered=[158,170,164,153,146,142,108,121,101]          # new -> cold
promoted=[30,27,33,30,36,28,28,42,40]                   # (cold->active)+(new->active)
quar=[0,0,1,0,0,0,0,0,1]                                 # ->quarantined
x=np.arange(len(rounds)); w=0.4
fig,ax=plt.subplots(figsize=(7.2,4.3))
ax.bar(x-w/2,entered,w,label="entered cold (new skills)",color="#9A9A9A",edgecolor="white",linewidth=0.6)
ax.bar(x+w/2,promoted,w,label="promoted cold→active",color="#0072B2",edgecolor="white",linewidth=0.6)
# quarantine markers
for i,q in enumerate(quar):
    if q: ax.scatter([x[i]+w/2],[promoted[i]+8],marker="v",color="#D55E00",s=40,zorder=5)
ax.set_xticks(x); ax.set_xticklabels([str(r) for r in rounds])
ax.set_xlabel("Self-play round (end of each 150-round window)")
ax.set_ylabel("Skills per window")
ax.set_title("Repair-skill lifecycle flows: steady promotion, intake tapers,\nquarantine almost never fires",fontsize=11,loc="left")
ax.grid(axis="x",visible=False); ax.legend(frameon=False,loc="upper right")
ax.text(0.985,0.60,"quarantined (▼): 2 skills\nin 1,350 rounds",transform=ax.transAxes,
        ha="right",va="top",fontsize=8.5,color="#D55E00",
        bbox=dict(boxstyle="round,pad=0.35",fc="#FBEEE6",ec="#E6C7B3"))
ax.set_ylim(0,200)
fig.tight_layout(); fig.savefig("fig_lifecycle_flow.pdf"); fig.savefig("fig_lifecycle_flow.png",dpi=150)
print("ok")
