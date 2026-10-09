import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OKABE = {"naked":"#999999","memory":"#0072B2","escal":"#009E73"}
SRC = ["human","edited_lm","qwen7b","gpt_oss"]
# per model: source -> (naked, memory, escalation)
data = {
 "glm-flash":  [(31.8,34.6,43.0),(44.9,40.9,51.6),(28.0,28.5,36.8),(18.8,25.3,30.0)],
 "nova-pro":   [(40.1,38.5,46.9),(49.7,49.7,57.0),(28.2,31.2,38.2),(23.1,27.8,33.8)],
 "gpt-5-nano": [(64.3,60.9,70.8),(73.5,69.8,78.6),(58.1,55.6,65.6),(60.0,48.1,63.4)],
 "gpt-5-mini": [(66.1,70.8,73.4),(78.1,77.0,82.9),(67.2,65.6,72.6),(47.5,44.4,51.2)],
}
order = ["glm-flash","nova-pro","gpt-5-nano","gpt-5-mini"]

fig, axes = plt.subplots(2,2, figsize=(10,6.4), sharey=True)
x = np.arange(len(SRC)); w = 0.26
for ax, model in zip(axes.flat, order):
    rows = data[model]
    nk=[r[0] for r in rows]; me=[r[1] for r in rows]; es=[r[2] for r in rows]
    ax.bar(x-w, nk, w, color=OKABE["naked"], label="naked")
    ax.bar(x,   me, w, color=OKABE["memory"], label="memory")
    ax.bar(x+w, es, w, color=OKABE["escal"], label="escalation")
    ax.set_title(model, fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(SRC, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 90)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
for ax in axes[:,0]:
    ax.set_ylabel("solved@$k$ (\\%)")
axes[0,0].legend(frameon=False, fontsize=8.5, loc="upper right")
plt.tight_layout()
plt.savefig("fig_e5_persource.pdf"); plt.savefig("fig_e5_persource.png", dpi=150)
print("wrote fig_e5_persource.pdf/.png")
