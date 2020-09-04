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
sns.set(context='paper', style='whitegrid', font_scale=6)

# datasets = ['D1', 'D2', 'cdip']
datasets = ['D1', 'D2']
template_fname = 'ablation/results/{}_0.2_1000_{}x64_128_fire3_1.0_0.1_3.json'
records = []
sizes = [32, 64, 128, 256]
for dataset, size in product(datasets, sizes):
    fname = template_fname.format(dataset, size)
    print(fname)

    results = json.load(open(fname, 'r'))['data']
    for run in results:
        accuracy = run['accuracy']
        #total_time = run['prep_time'] + run['pw_time'] + run['opt_time']
        records.append([dataset, size, 100 * accuracy])

df = pd.DataFrame.from_records(records, columns=('dataset', 'size', 'accuracy'))
print(df)

# new names for datasets
datasets_map = {'D1': '\\textsc{S-Marques}', 'D2': '\\textsc{S-Isri-OCR}', 'cdip': '\\textsc{S-cdip}'}
# # df['dataset'].replace(datasets_map, inplace=True)
# df.sort_values(by='d', inplace=True)
path = 'charts'
if len(sys.argv) > 1:
    path = sys.argv[1]

# # fp = sns.FacetGrid(col='dataset', hue='neutral_thresh', height=8, aspect=1.0, data=df, legend_out=False)
fp = sns.FacetGrid(hue='dataset', height=7, aspect=2.5, data=df, legend_out=True)
fp = (fp.map(sns.lineplot, 'size', 'accuracy', marker='s', ci=None, markersize=30).add_legend(title='\\textbf{dataset}',  prop={'size': 40}, labelspacing=0.25))

fp.ax.set_xscale('log', basex=2)
yticks = [90, 92.5, 95]
fp.set(yticks=yticks, xticks=sizes, ylim=(90, 96))
# fp.ax.legend_.set_title('\\textbf{dataset}')
fp.ax.set_xlabel('$\log_2(s_y)$', fontsize=60)
fp.ax.set_ylabel('accuracy (\\%)', fontsize=50)
fp.ax.set_xticklabels([str(int(x)) for x in np.log2(sizes)], fontdict={'fontsize': 50})
fp.ax.set_yticklabels([str(y) for y in yticks], fontdict={'fontsize': 50}) # trick to change the font

# legend
leg = fp._legend
leg.set_frame_on(True)
bb = leg.get_bbox_to_anchor().inverse_transformed(fp.ax.transAxes)
bb.x0 += 0.025
bb.x1 += 0.025
leg.set_bbox_to_anchor(bb, transform=fp.ax.transAxes)

plt.savefig('{}/ablation_input_size.pdf'.format(path), bbox_inches='tight')
#plt.show()
