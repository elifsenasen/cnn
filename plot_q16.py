
import json
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10})

RUNS = {"max": "q14_max_N81.json", "mean": "q14_mean_N81.json"}
COLORS = {"max": "#2a78d6", "mean": "#eb6834"}          
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e0"

results = {}
for pool, path in RUNS.items():
    with open(path) as f:
        results[pool] = json.load(f)


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


max_epochs = max(len(h["val_acc"]) for h in results.values())
ticks = list(range(1, max_epochs + 1, 2 if max_epochs > 12 else 1))

fig, ax = plt.subplots(figsize=(6, 3.6))
for pool, h in results.items():
    acc = [100 * a for a in h["val_acc"]]
    epochs = range(1, len(acc) + 1)
    ax.plot(epochs, acc, color=COLORS[pool], linewidth=2, marker="o", markersize=4,
            label=f"global {pool} pool")
    best = max(range(len(acc)), key=lambda i: acc[i])
    ax.plot(best + 1, acc[best], marker="o", markersize=10, markerfacecolor="none",
            markeredgecolor=COLORS[pool], markeredgewidth=1.5)

ax.set_xlabel("epoch")
ax.set_ylabel("validation accuracy (%)")
ax.set_xticks(ticks)
style(ax)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig("q16_val_acc.png", dpi=200)
fig.savefig("q16_val_acc.pdf")

fig, ax = plt.subplots(figsize=(6, 3.6))
for pool, h in results.items():
    epochs = range(1, len(h["val_loss"]) + 1)
    ax.plot(epochs, h["val_loss"], color=COLORS[pool], linewidth=2, marker="o", markersize=4,
            label=f"{pool}: validation")
    ax.plot(epochs, h["train_loss"], color=COLORS[pool], linewidth=1.5, linestyle="--",
            label=f"{pool}: training")

ax.set_yscale("log")                      
ax.set_xlabel("epoch")
ax.set_ylabel("cross-entropy loss (log scale)")
ax.set_xticks(ticks)
style(ax)
ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig("q16_loss.png", dpi=200)
fig.savefig("q16_loss.pdf")

print("saved q16_val_acc.png/.pdf and q16_loss.png/.pdf")
