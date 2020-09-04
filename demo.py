import os
import time
import json
import argparse
import random
import matplotlib.pyplot as plt

from docrec.metrics import accuracy
from docrec.strips.strips import Strips
from docrec.compatibility.proposed import Proposed
from docrec.pipeline import Pipeline
from docrec.solverconcorde import SolverConcorde


NUM_CLASSES = 2

# parameters processing
parser = argparse.ArgumentParser(description='Reconstruction demo.')
parser.add_argument(
    '-d', '--doc', action='store', dest='doc', required=False, type=str,
    default='datasets/D2/mechanical/D010', help='Instance to be tested.'
)
parser.add_argument(
    '-t', '--thresh', action='store', dest='thresh', required=False, type=str,
    default='sauvola', help='Thresholding method [otsu or sauvola].'
)
parser.add_argument(
    '-m', '--model-id', action='store', dest='model_id', required=False, type=str,
    default='cdip_0.2_1000_32x64_128_fire3_1.0_0.1', help='Model identifier (directory in traindata).'
)
parser.add_argument(
    '-i', '--input-size', action='store', dest='input_size', required=False, nargs=2, type=int,
    default=[3000, 64], help='Overall networks input size (H x W) for test: (H x W/2) for each network'
)
parser.add_argument(
    '-v', '--vshift', action='store', dest='vshift', required=False, type=int,
    default=10, help='Vertical shift range.'
)
parser.add_argument(
    '-fd', '--feat-dim', action='store', dest='feat_dim', required=False, type=int,
    default=128, help='Features dimensionality (d parameter in the paper).'
)
parser.add_argument(
    '-fl', '--feat-layer', action='store', dest='feat_layer', required=False, type=str,
    default='fire3', help='Features layer (from where features are extracted).'
)
parser.add_argument(
    '-a', '--activation', action='store', dest='activation', required=False, type=str,
    default='sigmoid', help='Activation function of the feature layer.'
)
args = parser.parse_args()

# assert args.solver in ['concorde', 'kbh', 'LS']
# assert args.thresh in ['otsu', 'sauvola']

# random.seed(int(args.seed))

# algorithm definition
weights_path_left = json.load(open('traindata/{}/info.json'.format(args.model_id), 'r'))['best_model_left']
weights_path_right = json.load(open('traindata/{}/info.json'.format(args.model_id), 'r'))['best_model_right']
sample_height = json.load(open('traindata/{}/info.json'.format(args.model_id), 'r'))['sample_height']
algorithm = Proposed(
    weights_path_left, weights_path_right, args.vshift,
    args.input_size, feat_dim=args.feat_dim, feat_layer=args.feat_layer,
    activation=args.activation, sample_height=sample_height,
    thresh_method=args.thresh
)
print(sample_height)

# pipeline: compatibility algorithm + solver
solver = SolverConcorde(maximize=False, max_precision=2)
pipeline = Pipeline(algorithm, solver)

# load strips and shuffle the strips
print('1) Load strips')
strips = Strips(path=args.doc, filter_blanks=True)
strips.shuffle()
init_permutation = strips.permutation()
print('Shuffled order: ' + str(init_permutation))

print('2) Results')
solution, compatibilities, displacements = pipeline.run(strips)
displacements = [displacements[prev][curr] for prev, curr in zip(solution[: -1], solution[1 :])]
corrected = [init_permutation[idx] for idx in solution]
print('Solution: ' + str(solution))
print('Correct order: ' + str(corrected))
print('Accuracy={:.2f}%'.format(100 * accuracy(solution, init_permutation)))
reconstruction = strips.image(order=solution, displacements=displacements)
plt.imshow(reconstruction, cmap='gray')
plt.axis('off')
plt.show()