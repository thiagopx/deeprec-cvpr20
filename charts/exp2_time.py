# https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/bar_stacked.html
import sys
import json
import numpy as np
import pandas as pd
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

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
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 3), sharey=True)

# 20 documents
N = len(results["solution"])  # total (20)
gray = [0.3] * 3
width = 0.35  # the width of the bars: can also be len(x) sequence
x1 = [0.55, 1.45]
print(matrix)
p1 = ax1.bar(x1, matrix[:, 0], width, color=colors[0], linewidth=1, edgecolor=gray)  # pro
p2 = ax1.bar(x1, matrix[:, 1], width, bottom=matrix[:, 0], color=colors[1], edgecolor=gray)  # pw
p3 = ax1.bar(x1, matrix[:, 2], width, bottom=matrix[:, 0] + matrix[:, 1], color=colors[2], edgecolor=gray)  # opt

y1_ticks = [60 * x for x in range(0, 81, 20)]
ax1.set_xlim([0, 2])
ax1.set_xlabel("method")
ax1.set_yticks(y1_ticks)
ax1.set_yticklabels([str(y // 60) for y in y1_ticks])
ax1.set_ylabel("time (min.)")
ax1.legend((p1[0], p2[0], p3[0]), ("pro", "pw", "opt"), title="\\textbf{stage}", fontsize=16, title_fontsize=16)
ax1.set_title("$n={}$".format(N))
ax1.set_xticks(x1)
ax1.set_xticklabels(("\\textbf{Deeprec-ML}", "Deeprec-CL"))

# n = 1, 2,3 ..., 1000
x2 = np.arange(10, N)
matrix_avg = np.array(
    [
        #      pro               pw
        [matrix[0, 0] / N, matrix[0, 1] / (N * (N - 1))],  # prop
        [matrix[1, 0] / N, matrix[1, 1] / (N * (N - 1))],  # base
    ]
)

y2_ticks = y1_ticks
# # p5 = ax2.bar(x, matrix[2 :, 1], width, bottom=matrix[2 :, 0], color=colors[1], edgecolor=gray) # pw
total_prop = matrix_avg[0, 0] * x2 + matrix_avg[0, 1] * (x2 * (x2 - 1))
total_sib18 = matrix_avg[1, 0] * x2 + matrix_avg[1, 1] * (x2 * (x2 - 1))
p4 = ax2.plot(x2, total_prop, color=colors[0])  # prop
p5 = ax2.plot(x2, total_sib18, color=colors[2])  # base

ax2.set_title("$n=1, 2, \ldots, {}$".format(N))
ax2.set_xticks(x2)
ax2.set_xticklabels([str(x) if x in [30, 100, 200, 300, 400, 500] else "" for x in x2])
ax2.set_xlabel("$n$")
ax2.set_yticks(y2_ticks)
ax2.legend(
    (p4[0], p5[0]), ("\\textbf{Deeprec-ML}", "Deeprec-CL"), title="\\textbf{method}", fontsize=16, title_fontsize=16
)

idx = np.where(x2 == 30)[0][0]
t_prop = total_prop[idx]
t_sib18 = total_sib18[idx]

ax2.annotate(
    "$\\approx$1 page", xy=(30, 10), xytext=(40, 1200), arrowprops=dict(facecolor="black", shrink=0.01), fontsize=14
)
# plt.draw() # calculate labels
for ax in [ax1, ax2]:
    ax.xaxis.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

plt.savefig("{}/exp2_time.pdf".format(path), bbox_inches="tight")
