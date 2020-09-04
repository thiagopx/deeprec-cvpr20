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
sns.set(
    context='paper', style='darkgrid', font_scale=6.75,
    rc={'font.serif': ['DejaVu Serif']}
)

datasets = ['D1', 'D2', 'cdip']
# template_fname = 'ablation/results_by_epoch/{}_0.2_1000_32x32_4_1.0_3.json'
template_fname = 'ablation/results_by_epoch/{}_0.2_1000_32x64_128_fire3_1.0_0.1_3.json'
# template_fname = 'ablation/results_by_epoch/{}_0.2_1000_32x32_128_1.0_3.json'

records = []
#feat_dim_range = [2, 4, 8, 16, 32, 64, 128, 256, 512]
for dataset in datasets:
    fname = template_fname.format(dataset)
    print(fname)

    results = json.load(open(fname, 'r'))['data']
    for run in results:
        epoch = run['epoch']
        accuracy = run['accuracy']
        records.append([dataset, epoch, 100 * accuracy])

print(len(records))
df = pd.DataFrame.from_records(records, columns=('dataset', 'epoch', 'accuracy (\\%)'))

# new names for datasets
datasets_map = {'D1': '\\textsc{S-Marques}', 'D2': '\\textsc{S-Isri-OCR}', 'cdip': '\\textsc{S-cdip}'}
df['dataset'].replace(datasets_map, inplace=True)
df.sort_values(by='epoch', inplace=True)
path = 'charts'
if len(sys.argv) > 1:
    path = sys.argv[1]

fp = sns.FacetGrid(hue='dataset', height=12, aspect=2.0, data=df, legend_out=False)
fp = (fp.map(sns.lineplot, 'epoch', 'accuracy (\\%)', marker='s', ci=None, markersize=20).add_legend(title='dataset',  prop={'size': 40}, labelspacing=0.25))

fp.set(yticks=(50, 60, 70, 80, 90, 100), xticks=df['epoch'].unique(), ylim=(50, 101))
# fp.ax.set_xscale('log', basex=2)
fp.set_yticklabels((50, 60, 70, 80, 90, 100))

mask = (np.arange(df['epoch'].unique().size) % 2).astype(np.bool)
xlabels = df['epoch'].unique().astype(np.str)
xlabels[mask] = ''
fp.set_xticklabels(xlabels)
# #ax.set(, yscale="log")
# print(np.log2(feat_dim_range))
# # g.set_yticklabels((40, 60, 80, 100), fontdict={'fontsize': 30})


plt.savefig('{}/ablation_epochs.pdf'.format(path), bbox_inches='tight')
# # # plt.show()
