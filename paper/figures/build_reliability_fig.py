import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
 "axes.grid":True,"grid.color":"#E6E6E6","grid.linewidth":0.8,"axes.axisbelow":True,
 "figure.dpi":140,"savefig.bbox":"tight","font.family":"DejaVu Sans"})
# averaged over 2 passes
naked =[260.0,121.5,178.0,73.5,92.0]
memory=[297.0,87.0,95.5,50.5,195.0]
x=np.arange(5); w=0.38
fig,ax=plt.subplots(figsize=(6.6,4.2))
b1=ax.bar(x-w/2,naked,w,label="naked",color="#9A9A9A",edgecolor="white",linewidth=0.6)
b2=ax.bar(x+w/2,memory,w,label="memory",color="#0072B2",edgecolor="white",linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels(["0","1","2","3","4"])
ax.set_xlabel("Attempts passed, out of 4  (reliability of the repair)")
ax.set_ylabel("Number of bugs")
ax.set_title("Memory makes fixes more reliable, not more numerous",fontsize=11,loc="left")
ax.grid(axis="x",visible=False); ax.legend(frameon=False,loc="upper center")
ax.annotate("more all-pass\n(reliable)",xy=(4+w/2,195),xytext=(3.1,250),fontsize=8.5,color="#0072B2",
            ha="center",arrowprops=dict(arrowstyle="->",color="#0072B2",lw=1))
ax.annotate("more all-fail",xy=(0+w/2,297),xytext=(0.9,320),fontsize=8.5,color="#555555",
            ha="center",arrowprops=dict(arrowstyle="->",color="#777777",lw=1))
ax.text(0.985,0.62,"mean solve-rate: naked 0.37, memory 0.42\n"
        "on the 773 bugs both solve:\nmemory 2.94 vs naked 2.44 passes",
        transform=ax.transAxes,ha="right",va="top",fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4",fc="#F2F2F2",ec="#CCCCCC"))
ax.set_ylim(0,340)
fig.tight_layout(); fig.savefig("fig_reliability.pdf"); fig.savefig("fig_reliability.png",dpi=150)
print("ok")
