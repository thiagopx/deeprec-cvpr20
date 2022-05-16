# https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/bar_stacked.html
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(
    context="paper",
    style="whitegrid",
    palette="muted",
    font_scale=2,
    rc={"text.usetex": True, "font.sans-serif": ["Arial"]},
)
colors = sns.color_palette("muted", n_colors=3)

path = "charts"
if len(sys.argv) > 1:
    path = sys.argv[1]

# proposed
results = json.load(open("results/proposed_multi/D2.json", "r"))["data"][0]
prop = np.array([results["prep_time"], results["pw_time"], results["opt_time"]])

# base work
results = json.load(open("results/sib18_multi/D2.json", "r"))["data"][0]
sib18 = np.array([results["prep_time"], results["pw_time"], results["opt_time"]])
matrix = np.array([prop, sib18])

# Set figsize here
fig, axd = plt.subplot_mosaic(
    [["A", "A", "B", "B", "B"]], figsize=(12, 3), constrained_layout=True
)
ax1, ax2 = axd["A"], axd["B"]
fig.tight_layout(pad=1.0)

# 20 documents
size_dataset = len(results["solution"])
gray = [0.3] * 3
width = 0.35  # the width of the bars: can also be len(x) sequence
x1 = [0.55, 1.45]
p1 = ax1.bar(
    x1, matrix[:, 0], width, color=colors[0], linewidth=1, edgecolor=gray
)  # pro
p2 = ax1.bar(
    x1, matrix[:, 1], width, bottom=matrix[:, 0], color=colors[1], edgecolor=gray
)  # pw
p3 = ax1.bar(
    x1,
    matrix[:, 2],
    width,
    bottom=matrix[:, 0] + matrix[:, 1],
    color=colors[2],
    edgecolor=gray,
)  # opt

y1_ticks = [60 * x for x in range(0, 81, 20)]
ax1.set_xlim([0, 2])
ax1.set_xlabel("method")
ax1.set_yticks(y1_ticks)
ax1.set_yticklabels([str(y // 60) for y in y1_ticks])
ax1.set_ylabel("time (min.)")
ax1.legend(
    (p1[0], p2[0], p3[0]),
    ("pro", "pw", "opt"),
    title="\\textbf{stage}",
    fontsize=16,
    title_fontsize=16,
)
ax1.set_title("$n={}$".format(size_dataset))
ax1.set_xticks(x1)
ax1.set_xticklabels(("\\textsc{Deeprec-ML}", "\\textsc{Deeprec-CL}"), fontsize=14)

matrix_avg = np.array(
    [
        [
            # proposed
            matrix[0, 0] / size_dataset,  # pro
            matrix[0, 1] / (size_dataset * (size_dataset - 1)),  # pw
        ],
        [
            # sib 18
            matrix[1, 0] / size_dataset,  # pro
            matrix[1, 1] / (size_dataset * (size_dataset - 1)),  # pw
        ],
    ]
)

# 100, 11, 12, ..., 505
size_total = 3000
x2 = np.arange(2, size_total + 1)
y2_ticks = [60 * x for x in range(0, 4001, 500)]
total_sib18 = matrix_avg[1, 0] * x2 + matrix_avg[1, 1] * (x2 * (x2 - 1))
total_prop = matrix_avg[0, 0] * x2 + matrix_avg[0, 1] * (x2 * (x2 - 1))
p4 = ax2.plot(x2, total_prop, color=colors[0])  # prop
p5 = ax2.plot(x2, total_sib18, color=colors[2])  # base
xticks = [30, size_dataset, 1000, 1500, 2000, 2500, 3000]
ax2.set_xticks(xticks)
ax2.set_xticklabels([str(x) for x in xticks])
ax2.set_xlabel("$n$")
ax2.set_xlim([0, 3000])
ax2.set_ylim([0, 66 * 3600])
ax2.set_yticks(y2_ticks)
ax2.set_yticklabels([str(y // 3600) for y in y2_ticks])
ax2.set_ylabel("time (hour)")
ax2.vlines(size_dataset, 0, 66 * 3600, linestyles="dashed", colors="red")
ax2.legend(
    (p4[0], p5[0]),
    ("\\textsc{Deeprec-ML}", "\\textsc{Deeprec-CL}"),
    title="\\textbf{method}",
    fontsize=14,
    title_fontsize=16,
    loc="upper center",
)

ax2.annotate(
    "$\\approx$1 page",
    xy=(30, 10),
    xytext=(80, 60000),
    arrowprops=dict(facecolor="black", shrink=0.01),
    fontsize=14,
)

ax2.annotate(
    "$\\approx$22x of speed-up",
    xy=(505, 10000),
    xytext=(650, 60000),
    arrowprops=dict(facecolor="black", shrink=0.01),
    fontsize=14,
)

# # # plt.draw() # calculate labels
for ax in [ax1, ax2]:
    ax.xaxis.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

print("speed-up(505) (real) ={:.3f}".format(sib18[:2].sum() / prop[:2].sum()))
print("speed-up(505) (est) = {:.3f}".format(total_sib18[504] / total_prop[504]))
print("speed-up(3000) = {:.3f}".format(total_sib18[-1] / total_prop[-1]))
plt.savefig("{}/exp2_time_extended.pdf".format(path))  # , bbox_inches="tight")

print(total_sib18[:-10] / total_prop[:-10])
