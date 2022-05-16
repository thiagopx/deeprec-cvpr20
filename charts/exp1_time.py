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

datasets = ["D1", "D2"]  # , 'cdip']
template_fname_prop = "results/proposed/{}_0.2_1000_32x64_128_fire3_1.0_0.1_{}.json"
template_fname_sib18 = "results/sib18/{}-{}.json"
# records = {('D1', 'off'): [], ('D1', 'on'): [], ('D2', 'off'): [], ('D2', 'on'): []}
records = []
for dataset in datasets:
    for vshift_prop, vshift_sib18, enabled in zip([0, 3], [0, 10], ["off", "on"]):
        for method, vshift, template_fname in zip(
            ["proposed", "sib18"],
            [vshift_prop, vshift_sib18],
            [template_fname_prop, template_fname_sib18],
        ):
            fname = template_fname.format(dataset, vshift)
            results = json.load(open(fname, "r"))["data"]
            for run in results:
                doc = run["doc"].split("/")[-1]
                records.append(
                    [
                        method,
                        dataset,
                        doc,
                        enabled,
                        len(run["solution"]),
                        run["prep_time"],
                        run["pw_time"],
                        run["opt_time"],
                    ]
                )


df = pd.DataFrame.from_records(
    records, columns=("method", "dataset", "doc", "enabled", "n", "pro", "pw", "opt")
)
print(df)

# Set figsize here
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 3))

width = 0.5  # the width of the bars: can also be len(x) sequence
x = [0.0, 1.5]

matrix = []
summary = df.groupby(["method", "enabled"])["pro", "pw", "opt"].mean()
print(summary.index)
print(summary.columns)
print(summary.head())

for enabled in ["off", "on"]:
    for method in ["proposed", "sib18"]:
        matrix.append(summary.loc[(method, enabled)].values)

matrix = np.array(matrix)
print(matrix)
gray = [0.3] * 3
p1 = ax1.bar(
    x, matrix[:2, 0], width, color=colors[0], linewidth=1, edgecolor=gray
)  # pro
p2 = ax1.bar(
    x, matrix[:2, 1], width, bottom=matrix[:2, 0], color=colors[1], edgecolor=gray
)  # pw
p3 = ax1.bar(
    x,
    matrix[:2, 2],
    width,
    bottom=matrix[:2, 0] + matrix[:2, 1],
    color=colors[2],
    edgecolor=gray,
)  # opt

p4 = ax2.bar(
    x, matrix[2:, 0], width, color=colors[0], linewidth=1, edgecolor=gray
)  # pro
p5 = ax2.bar(
    x, matrix[2:, 1], width, bottom=matrix[2:, 0], color=colors[1], edgecolor=gray
)  # pw
p6 = ax2.bar(
    x,
    matrix[2:, 2],
    width,
    bottom=matrix[2:, 0] + matrix[2:, 1],
    color=colors[2],
    edgecolor=gray,
)  # opt

ax1.set_ylabel("time (sec.)")
# # plt.yticks(np.arange(0, 81, 10))
ax2.legend(
    (p1[0], p2[0], p3[0]),
    ("pro", "pw", "opt"),
    title="\\textbf{stage}",
    fontsize=16,
    title_fontsize=16,
)
plt.draw()
y1_ticks = [0, 0.5, 1.0, 1.5]
y2_ticks = [0, 2.5, 5.0, 7.5, 10]
for ax, title, y_ticks in zip(
    [ax1, ax2], ["vertical shift (off)", "vertical shift (on)"], [y1_ticks, y2_ticks]
):
    ax.xaxis.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(x[0] - width, x[1] + width)
    ax.set_xlabel("method")
    ax.set_xticks(x)
    ax.set_xticklabels(("\\textsc{Deeprec-ML}", "\\textsc{Deeprec-CL}"), fontsize=18)
    ax.set_title(title)
    # print([label.get_text() for label in ax.get_yticklabels()])
    # ax.set_yticklabels([label.get_text().replace('$', '') for label in ax.get_yticklabels()])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks])


plt.savefig("{}/exp1_time.pdf".format(path), bbox_inches="tight")
