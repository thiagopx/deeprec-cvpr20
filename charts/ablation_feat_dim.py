import sys
import json
from itertools import product
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from matplotlib import rc
rc('text', usetex=True)

import seaborn as sns
sns.set(context='paper', style='whitegrid', font_scale=2)

datasets = ['D1', 'D2']
template_fname = 'results/proposed/{}_0.2_1000_32x64_{}_fire3_1.0_0.1_3.json'
records = []
feat_dim_range = [2, 4, 8, 16, 32, 64, 128, 256, 512]
for dataset, feat_dim in product(datasets, feat_dim_range):
    fname = template_fname.format(dataset, feat_dim)
    print(fname)

    results = json.load(open(fname, 'r'))['data']
    for run in results:
        accuracy = run['accuracy']
        total_time = run['prep_time'] + run['pw_time'] + run['opt_time']
        records.append([dataset, feat_dim, total_time, 100 * accuracy])

df = pd.DataFrame.from_records(records, columns=('dataset', 'd', 'time', 'accuracy (\\%)'))

# new names for datasets
# datasets_map = {'D1': '\\textsc{S-Marques}', 'D2': '\\textsc{S-Isri-OCR}', 'cdip': '\\textsc{S-cdip}'}
# df['dataset'].replace(datasets_map, inplace=True)
df.sort_values(by='d', inplace=True)
path = 'charts'
if len(sys.argv) > 1:
    path = sys.argv[1]

# fp = sns.FacetGrid(col='dataset', hue='neutral_thresh', height=8, aspect=1.0, data=df, legend_out=False)
fp = sns.FacetGrid(hue='dataset', height=4, aspect=2, data=df, legend_out=False)
fp = (fp.map(sns.lineplot, 'd', 'accuracy (\\%)', marker='s', ci=None, markersize=8).add_legend(title='dataset',  prop={'size': 16}, labelspacing=0.25))

# fp.ax.set_xscale('log', basex=2)
yticks = [80, 85, 90, 95, 100]
fp.ax.legend_.set_title('\\textbf{dataset}')
fp.ax.set_xlabel('$\log_2(d)$')
fp.ax.set(yticks=yticks, ylim=(80, 100))
#fp.ax.set_xticks(feat_dim_range)
# fp.ax.set_xticklabels([str(int(x)) for x in np.log2(feat_dim_range)])
fp.ax.xaxis.grid(False)
fp.ax.spines['left'].set_visible(False)
fp.ax.spines['top'].set_visible(False)
fp.ax.spines['right'].set_visible(False)
fp.ax.set_yticklabels([str(y) for y in yticks])

df_ = df.groupby(['d']).mean().reset_index()
print(df_)
ax2 = fp.ax.twinx()
ax2.set_xscale('log', basex=2)
ax2.set_ylabel('time (s)')
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
ax2.xaxis.grid(False)
ax2.yaxis.grid(False)
l,  = ax2.plot(feat_dim_range, df_['time'], '--', color='gray')
ax2.set(xticks=feat_dim_range, yticks=[1.5, 2.0, 2.5], ylim=(0.7, 2.75))
ax2.set_xticklabels([str(int(x)) for x in np.log2(feat_dim_range)])
ax2.set_yticklabels(str(x) for x in [1.5, 2.0, 2.5])
ax2.legend([l], ['proc. time'], loc=4)
plt.draw()

print(df_['time'].values)
# fp = (fp.map(sns.lineplot, 'd', 'accuracy (\\%)', style='--', ci=None, ax=ax2).add_legend(title='dataset', prop={'size': 16}, labelspacing=0.25))

# print(np.log2(feat_dim_range))
# g.set_yticklabels((40, 60, 80, 100), fontdict={'fontsize': 30})


plt.savefig('{}/ablation_feat_dim.pdf'.format(path), bbox_inches='tight')
# # plt.show()
