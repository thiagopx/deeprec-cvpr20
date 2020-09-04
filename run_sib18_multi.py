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
from docrec.compatibility.sib18 import Sib18
from docrec.solverconcorde import SolverConcorde
from docrec.pipeline import Pipeline


def reconstruct():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # parameters processing
    parser = argparse.ArgumentParser(description='Testing reconstruction of mixed documents.')
    parser.add_argument(
        '-d', '--dataset', action='store', dest='dataset', required=False, type=str,
        default='cdip', help='Dataset [D1, D2, or cdip].'
    )
    parser.add_argument(
        '-r', '--results-id', action='store', dest='results_id', required=False, type=str,
        default=None, help='Identifier of the results file.'
    )
    parser.add_argument(
        '-v', '--vshift', action='store', dest='vshift', required=False, type=int,
        default=10, help='Vertical shift range.'
    )

    args = parser.parse_args()

    assert args.dataset in ['D1', 'D2', 'D1+D2', 'cdip']
    assert args.results_id is not None

    # algorithm definition
    weights_path = json.load(open('traindata/sib18/info.json', 'r'))['best_model']
    algorithm = Sib18(
        'sn', weights_path, args.vshift, (3000, 32), num_classes=2,
        thresh_method='sauvola', seed=SEED, sess=sess
    )
    solver = SolverConcorde(maximize=True, max_precision=2)
    pipeline = Pipeline(algorithm, solver)

    # reconstruction instances
    if args.dataset == 'D1':
        docs = ['datasets/D1/mechanical/D{:03}'.format(i) for i in range(1, 62) if i != 3]
    elif args.dataset == 'D2':
        docs = ['datasets/D2/mechanical/D{:03}'.format(i) for i in range(1, 21)]
    else:
        docs = ['datasets/D3/mechanical/D{:03}'.format(i) for i in range(1, 101)] # cdip

    # results
    dir_name = 'results/sib18_multi'
    os.makedirs(dir_name, exist_ok=True)
    results_fname = '{}/{}.json'.format(dir_name, args.results_id)
    compatibility_matrix_fname = '{}/{}.npy'.format(dir_name, args.results_id)
    results = {
        'model_id': 'sib18',
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
