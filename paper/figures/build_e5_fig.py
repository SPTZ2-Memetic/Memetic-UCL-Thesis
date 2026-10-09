import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito (consistent with earlier figures)
OKABE = {"naked":"#999999", "memory":"#0072B2", "escal":"#009E73"}

# fixers ordered by naked solved@k (capability axis); 2-pass means, mini=pass1 only
fixers = ["glm-flash", "nova-pro", "gpt-5-nano", "gpt-5-mini"]
naked  = [31.5, 35.5, 64.1, 65.0]
memory = [32.5, 37.0, 59.0, 65.0]
escal  = [41.0, 44.5, 69.9, 71.0]

x = np.arange(len(fixers)); w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.bar(x - w, naked,  w, label="naked",              color=OKABE["naked"])
ax.bar(x,     memory, w, label="memory (uncond.)",   color=OKABE["memory"])
ax.bar(x + w, escal,  w, label="escalation",         color=OKABE["escal"])

for i in range(len(fixers)):
    d_mem = memory[i]-naked[i]; d_esc = escal[i]-naked[i]
    ax.text(x[i],     memory[i]+1.2, f"{d_mem:+.0f}", ha="center", va="bottom", fontsize=8, color=OKABE["memory"])
    ax.text(x[i]+w,   escal[i]+1.2,  f"{d_esc:+.0f}", ha="center", va="bottom", fontsize=8, color=OKABE["escal"])

ax.set_xticks(x); ax.set_xticklabels(fixers)
ax.set_ylabel("solved@$k$ (\\%)"); ax.set_ylim(0, 82)
ax.set_xlabel("fixer, ordered by naked solve rate (increasing capability)")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.spines[["top","right"]].set_visible(False)
ax.axhline(0, color="black", lw=0.5)
plt.tight_layout()
plt.savefig("fig_e5_capability.pdf"); plt.savefig("fig_e5_capability.png", dpi=150)
print("wrote fig_e5_capability.pdf / .png")

# print the delta summary for the prose/table
print("\nfixer        naked  mem  esc   dMem   dEsc")
for i,f in enumerate(fixers):
    print(f"{f:12} {naked[i]:5.1f} {memory[i]:4.1f} {escal[i]:4.1f}  {memory[i]-naked[i]:+5.1f}  {escal[i]-naked[i]:+5.1f}")
