import sys
import json
from itertools import product
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from matplotlib import rc
rc('text', usetex=True)

import seaborn as sns
sns.set(context='paper', style='darkgrid', font_scale=4)
# sns.set_style({'font.family': ['sans-serif'], 'sans-serif': ['Arial']})

annotation = json.load(open('datasets/D3/mechanical/annotation.json'))
map_doc_category = {doc: annotation[doc]['category'] for doc in annotation}

template_fname = 'ablation/results/cdip_0.2_1000_{}x{}_10.json'
records = []
sizes = [32, 64]

for H, W in product(sizes, sizes):
    fname = template_fname.format(H, W)
    print(fname)
    runs = json.load(open(fname, 'r'))['data']['1']
    for run in runs:
        doc = run['docs'][0].split('/')[-1]
        category = map_doc_category[doc]
        accuracy = run['accuracy']
        records.append([H, W, '{} $\\times$ {}'.format(H, W), category, 100 * accuracy])

df = pd.DataFrame.from_records(records, columns=('H', 'W', 'size', 'Category', 'Accuracy (\\%)'))
map_category = {
    'news_article': 'news article'
}
df['Category'].replace(map_category, inplace=True)
# new names for datasets
# datasets_map = {'D1': '\\textsc{S-Marques}', 'D2': '\\textsc{S-Isri-OCR}', 'cdip': '\\textsc{S-cdip}'}
# df['dataset'].replace(datasets_map, inplace=True)
df.sort_values(by='size', inplace=True)
path = 'charts'
if len(sys.argv) > 1:
    path = sys.argv[1]

# max_value = len(df['$k$'].unique())

# fp = sns.FacetGrid(hue='size', data=df, height=12, aspect=1., legend_out=True)
# font = font_manager.FontProperties(family='sans-serif', size=20)
# fp = (fp.map(sns.lineplot, 'category', 'Accuracy (\\%)', marker='s', ci=None, markersize=15).add_legend(title='sample size', prop={'size': 35}, labelspacing=0.7))
# fp.set(yticks=(75, 80, 85, 90, 95, 100), xticks=list(range(1, max_value + 1)), ylim=(75, 101))
g = sns.catplot(x='Category', y='Accuracy (\\%)', hue='size', data=df, kind='box', height=8, aspect=4)
g.set_xticklabels(rotation=30)
g.set(yticks=(40, 60, 80, 100), ylim=(40, 101))
# fp.set_xticklabels(list(range(1, max_value + 1)), fontdict={'fontsize': 35})

g.set_yticklabels((40, 60, 80, 100), fontdict={'fontsize': 30})
# g.set_yticklabels((40, 60, 80, 100), fontdict={'fontsize': 20})
#print(fp.axes)
#plt.setp(fp.axes[0, 0].get_legend().get_texts(), fontsize='8') # for legend text
plt.savefig('{}/ablation_input_size_cdip.pdf'.format(path), bbox_inches='tight')
plt.show()
