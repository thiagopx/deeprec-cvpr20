import os
import sys
import json
import numpy as np
import time
import argparse
import tensorflow as tf

# seed experiment
import random
SEED = 0
random.seed(SEED)
tf.set_random_seed(SEED)

from docrec.metrics import accuracy
from docrec.strips import Strips
from docrec.compatibility.proposed import Proposed
from docrec.solverconcorde import SolverConcorde
from docrec.pipeline import Pipeline

def reconstruct():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # parameters processing
    parser = argparse.ArgumentParser(description='Multi reconstruction :: Proposed.')
    parser.add_argument(
        '-d', '--dataset', action='store', dest='dataset', required=False, type=str,
        default='cdip', help='Dataset [D1, D2, D1+D2, cdip].'
    )
    parser.add_argument(
        '-t', '--thresh', action='store', dest='thresh', required=False, type=str,
        default='sauvola', help='Thresholding method [otsu or sauvola].'
    )
    parser.add_argument(
        '-m', '--model-id', action='store', dest='model_id', required=False, type=str,
        default=None, help='Model identifier (tag).'
    )
    parser.add_argument(
        '-i', '--input-size', action='store', dest='input_size', required=False, nargs=2, type=int,
        default=[3000, 64], help='Network input size (H x W).'
    )
    parser.add_argument(
       '-v', '--vshift', action='store', dest='vshift', required=False, type=int,
       default=3, help='Vertical shift range.'
    )
    parser.add_argument(
        '-r', '--results-id', action='store', dest='results_id', required=False, type=str,
        default=None, help='Identifier of the results file.'
    )
    parser.add_argument(
        '-fd', '--feat-dim', action='store', dest='feat_dim', required=False, type=int,
        default=64, help='Features dimensionality.'
    )
    parser.add_argument(
        '-fl', '--feat-layer', action='store', dest='feat_layer', required=False, type=str,
        default='drop9', help='Features layer.'
    )
    parser.add_argument(
        '-a', '--activation', action='store', dest='activation', required=False, type=str,
        default='sigmoid', help='Activation function (final net layer).'
    )

    args = parser.parse_args()

    input_size = tuple(args.input_size)

    assert args.dataset in ['D1', 'D2', 'D1+D2', 'cdip']
    assert args.thresh in ['otsu', 'sauvola']
    # assert args.results_id is not None
    assert args.vshift >= 0
    assert input_size in [(3000, 64)]

    # algorithm definition
    weights_path_left = json.load(open('traindata/{}/info.json'.format(args.model_id), 'r'))['best_model_left']
    weights_path_right = json.load(open('traindata/{}/info.json'.format(args.model_id), 'r'))['best_model_right']
    sample_height = json.load(open('traindata/{}/info.json'.format(args.model_id), 'r'))['sample_height']
    algorithm = Proposed(
        weights_path_left, weights_path_right, args.vshift,
        args.input_size, feat_dim=args.feat_dim, feat_layer=args.feat_layer,
        activation=args.activation, sample_height=sample_height,
        thresh_method=args.thresh, sess=sess
    )
    solver = SolverConcorde(maximize=False, max_precision=2)
    pipeline = Pipeline(algorithm, solver)

    # reconstruction instances
    docs1 = ['datasets/D1/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
    docs2 = ['datasets/D2/mechanical/D{:03}'.format(i) for i in range(1, 21)]
    docs3 = ['datasets/D3/mechanical/D{:03}'.format(i) for i in range(1, 101)] # cdip
    if args.dataset == 'D1+D2':
        docs = docs1 + docs2
    elif args.dataset == 'D1':
        docs = docs1
    elif args.dataset == 'D2':
        docs = docs2
    else:
        docs = docs3

    # results
    dir_name = 'results/proposed_multi'
    os.makedirs(dir_name, exist_ok=True)
    results_fname = '{}/{}.json'.format(dir_name, args.results_id)
    compatibility_matrix_fname = '{}/{}.npy'.format(dir_name, args.results_id)
    results = {
        'model_id': args.model_id,
        'data': [] # experimental results data
    }

    # load strips
    print('loading strips')
    t0 = time.time()
    strips = Strips() # empty
    for doc in docs: # join documents strips
        strips += Strips(path=doc, filter_blanks=True)
    strips.shuffle()
    load_time = time.time() - t0

    # run the pipeline
    print('running the pipeline')
    solution, compatibilities, displacements = pipeline.run(strips, verbose=True)

    # results
    acc = accuracy(solution, strips.permutation(), strips.sizes())

    print('accuracy={:.2f}% load_time={:.2f}s inf_time={:.2f}s comp_time={:.2f}s opt_time={:.2f}s'.format(
        100 * acc, load_time, algorithm.inference_time, pipeline.comp_time, pipeline.opt_time
    ))

    sys.stdout.flush()
    results['data'].append({
        'solution': solution,
        'accuracy': acc,
        'init_perm': strips.permutation(),
        'sizes': strips.sizes(),
        'load_time': load_time,
        'comp_time': pipeline.comp_time,
        'opt_time': pipeline.opt_time,
        'displacements': displacements.tolist(),
        'inf_time': algorithm.inference_time,
        'prep_time': algorithm.preparation_time,
        'pw_time': algorithm.pairwise_time
    })
    sess.close()

    # dump results and comp. matrix
    json.dump(results, open(results_fname, 'w'))
    np.save(compatibility_matrix_fname, compatibilities)


if __name__ == '__main__':
    t0 = time.time()
    reconstruct()
    t1 = time.time()
    print('Elapsed time={:.2f} minutes ({:.2f} seconds)'.format((t1 - t0) / 60.0, t1 - t0))
