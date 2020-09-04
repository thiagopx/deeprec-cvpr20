import sys
import json
import numpy as np
import pandas as pd
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

import seaborn as sns
sns.set(context='paper', style='whitegrid', palette='muted', font_scale=2,
    rc={
        'text.usetex' : True,
        'font.sans-serif': ['Arial']
    }
)
path = 'charts'
if len(sys.argv) > 1:
    path = sys.argv[1]

datasets = ['D1', 'D2']#, 'cdip']
template_fname_prop = 'results/proposed/{}_0.2_1000_32x64_128_fire3_1.0_0.1_{}.json'
template_fname_sib18 = 'results/sib18/{}-{}.json'
records = []
for dataset in datasets:
    for vshift_prop, vshift_sib18, enabled in zip([0, 3], [0, 10], ['off', 'on']):
        for method, vshift, template_fname in zip(['Proposed', 'Paix\~ao-b'], [vshift_prop, vshift_sib18], [template_fname_prop, template_fname_sib18]):
            fname = template_fname.format(dataset, vshift)
            results = json.load(open(fname, 'r'))['data']

            for run in results:
                accuracy = run['accuracy']
                comp_time = run['comp_time']
                inf_time = run['inf_time']
                opt_time = run['opt_time']
                pw_time = run['pw_time']
                prep_time = run['opt_time']
                doc = run['doc'].split('/')[-1]
                records.append([method, dataset, doc, enabled, 100 * accuracy, comp_time, inf_time, prep_time, pw_time, opt_time])

df = pd.DataFrame.from_records(records, columns=('method', 'dataset', 'doc', 'vert. shift', 'accuracy (\\%)', 'comp_time', 'inf_time', 'prep_time', 'pw_time', 'opt_time'))

print(df.groupby(['dataset', 'method', 'vert. shift']).mean())

meanlineprops = dict(linestyle='--', linewidth=1, color=(0.9, 0.0, 0.0))
meansqprops = {'marker': 's', 'markerfacecolor': 'white', 'markeredgecolor': 'blue'}

fp = sns.catplot(x ='vert. shift', y='accuracy (\\%)', col='dataset', hue='method', data=df, kind='box', height=4, aspect=1,
    margin_titles=True, fliersize=1.0, width=0.6, linewidth=1,
    legend=True, legend_out=False, showmeans=True, meanline=True,
    meanprops=meanlineprops, palette=['b', 'g']
)

ax1, ax2 = fp.axes.ravel()
ax1.legend_.parent = ax2
ax2.legend_ = ax1.legend_
ax2.legend_.set_title('\\textbf{method}')
yticks = [y for y in range(0, 101, 20)]
fp.set(yticks=yticks, ylim=(0, 105))

#ylabels = [y.get_text().replace('$', '') for y in ax1.get_yticklabels()]
# ax1.set_yticklabels(ylabels)
ax1.set_yticklabels(yticks)
for ax in fp.axes.ravel():
    ax.set_title(ax.get_title().replace('dataset = ', ''))
    ax.set_xlabel('vertical shift')
fp.despine(left=True)
plt.savefig('{}/exp1.pdf'.format(path), bbox_inches='tight')