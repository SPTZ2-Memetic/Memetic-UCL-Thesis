import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- verified aggregates (user ran extraction on runs/e3_*/{metrics,skillbank}.json) ----
# order within each sweep
# exploration sweep (frontier target = 0.5), x = skilled:exploration
expl = {
 "labels":   ["4:0","3:1","2:2","1:3"],
 "nskilled": [4,3,2,1],
 "valid":    [885,692,519,372],
 "malf_pct": [14.0,31.3,48.2,61.6],
 "fcount":   [149,135,103,76],
 "frate":    [0.1684,0.1951,0.1985,0.2043],
 "msr":      [0.8997,0.8887,0.8887,0.8595],
 "new":      [88,78,57,52],
 "H":        [2.183,2.059,2.071,2.023],
 "baseidx":  1,   # 3:1
}
# EMA sweep (ratio = 3:1), x = frontier target; 0.5 point IS expl_3_1
ema = {
 "labels":   ["0.3","0.5","0.7"],
 "valid":    [693,692,714],
 "malf_pct": [29.9,31.3,28.8],
 "fcount":   [142,135,124],
 "frate":    [0.2049,0.1951,0.1737],
 "msr":      [0.8763,0.8887,0.8824],
 "new":      [87,78,71],
 "H":        [2.081,2.059,2.124],
 "baseidx":  1,   # 0.5
}

BLUE="#0072B2"; ORANGE="#E69F00"; GREY="#999999"
def bars(ax, labels, vals, baseidx, title, ylab, fmt="{:.0f}", ylim=None):
    cols=[BLUE]*len(vals); cols[baseidx]=ORANGE
    b=ax.bar(range(len(vals)), vals, color=cols, width=0.62, zorder=3)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=10.5, pad=6)
    ax.set_ylabel(ylab, fontsize=9)
    ax.grid(axis="y", color="#e6e6e6", zorder=0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    if ylim: ax.set_ylim(*ylim)
    for i,(rect,v) in enumerate(zip(b,vals)):
        ax.text(rect.get_x()+rect.get_width()/2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=8.2,
                fontweight=("bold" if i==baseidx else "normal"))
    return b

fig, axs = plt.subplots(2,4, figsize=(14,6.6))
# Row 0: exploration sweep
r=expl
bars(axs[0,0], r["labels"], r["frate"], r["baseidx"], "Frontier rate", "frontier / valid", "{:.3f}", (0,0.24))
bars(axs[0,1], r["labels"], r["valid"], r["baseidx"], "Valid bugs produced", "count (of 300 rounds)", "{:.0f}", (0,980))
bars(axs[0,2], r["labels"], r["new"],   r["baseidx"], "New generator skills", "final bank − 196 seed", "{:.0f}", (0,98))
bars(axs[0,3], r["labels"], r["H"],     r["baseidx"], "Skill-bank diversity", "site-type entropy (bits)", "{:.2f}", (0,2.6))
# Row 1: EMA sweep
r=ema
bars(axs[1,0], r["labels"], r["frate"], r["baseidx"], "Frontier rate", "frontier / valid", "{:.3f}", (0,0.24))
bars(axs[1,1], r["labels"], r["valid"], r["baseidx"], "Valid bugs produced", "count (of 300 rounds)", "{:.0f}", (0,980))
bars(axs[1,2], r["labels"], r["new"],   r["baseidx"], "New generator skills", "final bank − 196 seed", "{:.0f}", (0,98))
bars(axs[1,3], r["labels"], r["H"],     r["baseidx"], "Skill-bank diversity", "site-type entropy (bits)", "{:.2f}", (0,2.6))

axs[0,0].annotate("Exploration–exploitation sweep\n(skilled : exploration, target 0.5)",
                  xy=(-0.42,0.5), xycoords="axes fraction", rotation=90,
                  va="center", ha="center", fontsize=10, fontweight="bold")
axs[1,0].annotate("Difficulty-target sweep\n(frontier target, ratio 3:1)",
                  xy=(-0.42,0.5), xycoords="axes fraction", rotation=90,
                  va="center", ha="center", fontsize=10, fontweight="bold")
# x labels
for ax in axs[0]: ax.set_xlabel("skilled : exploration", fontsize=8.5)
for ax in axs[1]: ax.set_xlabel("frontier target $\\rho^\\star$", fontsize=8.5)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(fc=ORANGE,label="baseline (3:1, $\\rho^\\star$=0.5)"),
                    Patch(fc=BLUE,label="ablation variant")],
           loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5,1.005))
fig.tight_layout(rect=[0.03,0,1,0.96])
fig.savefig("fig_ablation.pdf", bbox_inches="tight")
fig.savefig("fig_ablation.png", dpi=150, bbox_inches="tight")
print("wrote fig_ablation.pdf / .png")
