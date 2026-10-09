import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
 "axes.grid":True,"grid.color":"#E6E6E6","grid.linewidth":0.8,"axes.axisbelow":True,
 "figure.dpi":140,"savefig.bbox":"tight","font.family":"DejaVu Sans"})
labels=["0.0","0.25","0.5","0.75","1.0"]
vals=[211,93,174,266,1933]
tot=sum(vals)
# zone colours: too-hard (rho=0) muted, frontier blue, too-easy (rho=1) grey
colors=["#9A9A9A","#0072B2","#0072B2","#0072B2","#9A9A9A"]
fig,ax=plt.subplots(figsize=(6.8,4.2))
bars=ax.bar(range(5),vals,color=colors,width=0.72,edgecolor="white",linewidth=0.6)
for i,v in enumerate(vals):
    ax.text(i,v+25,f"{v}\n{100*v/tot:.0f}%",ha="center",va="bottom",fontsize=8.5,color="#333333")
ax.set_xticks(range(5)); ax.set_xticklabels(labels)
ax.set_xlabel("Per-bug solve-rate  $\\rho$  (fraction of 4 fixer attempts that pass)")
ax.set_ylabel("Number of valid generated bugs")
ax.set_title("Most generated bugs are easy for the fixer;\n20% sit at the learnable frontier",fontsize=11,loc="left")
ax.grid(axis="x",visible=False)
ax.set_ylim(0,2200)
# zone brackets / labels
ax.text(2,  560,"frontier  0<$\\rho$<1   (20%)",ha="center",fontsize=9.5,color="#0072B2")
fig.tight_layout(); fig.savefig("fig_difficulty.pdf"); fig.savefig("fig_difficulty.png",dpi=150)
print("ok")
