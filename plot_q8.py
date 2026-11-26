
import json
import matplotlib.pyplot as plt

with open("q8_results.json") as f:
    results = json.load(f)                      

lrs = sorted(results, key=float)


LR_COLORS = ["#8baad1", "#1772e1", "#0c2d55", "#070d14"]
TRAIN_C, VAL_C = "#ea5d26", "#2a78d6"           
INK, MUTED, GRID = "#171717", "#555252", "#d4d4b3"


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


def label(lr):
    return f"lr = {float(lr):g}"


# ---------------------------------------------------------------- validation accuracy
fig, ax = plt.subplots(figsize=(6, 3.6))
for lr, color in zip(lrs, LR_COLORS):
    acc = [100 * a for a in results[lr]["val_acc"]]
    epochs = range(1, len(acc) + 1)
    ax.plot(epochs, acc, color=color, linewidth=2, marker="o", markersize=5, label=label(lr))
    # hollow ring on the best epoch of each run
    best = max(range(len(acc)), key=lambda i: acc[i])
    ax.plot(best + 1, acc[best], marker="o", markersize=10, markerfacecolor="none",
            markeredgecolor=color, markeredgewidth=1.5)

ax.set_xlabel("epoch")
ax.set_ylabel("validation accuracy (%)")
ax.set_xticks(range(1, len(results[lrs[0]]["val_acc"]) + 1))
style(ax)
leg = ax.legend(title="learning rate (ring = best epoch)", frameon=False, fontsize=9,
                title_fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig("q8_val_acc.pdf")
fig.savefig("q8_val_acc.png", dpi=200)

# ---------------------------------------------------------------- train vs validation loss
fig, axes = plt.subplots(1, len(lrs), figsize=(12, 3.2), sharey=True)
for i, (ax, lr) in enumerate(zip(axes, lrs)):
    h = results[lr]
    epochs = range(1, len(h["train_loss"]) + 1)
    ax.plot(epochs, h["train_loss"], color=TRAIN_C, linewidth=2, marker="o", markersize=4, label="train")
    ax.plot(epochs, h["val_loss"], color=VAL_C, linewidth=2, marker="o", markersize=4, label="validation")
    ax.set_title(label(lr), fontsize=10, color=INK)
    ax.set_xlabel("epoch")
    ax.set_xticks(list(epochs)[::2] if len(epochs) > 6 else list(epochs))
    style(ax)
    # log scale: the first-epoch train loss (up to 0.5) would otherwise flatten the rest
    ax.set_yscale("log")
    if i == 0:
        ax.set_ylabel("cross-entropy loss (log scale)")
        ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig("q8_loss.pdf")
fig.savefig("q8_loss.png", dpi=200)

print("saved q8_val_acc.pdf/.png and q8_loss.pdf/.png")
