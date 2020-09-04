import os
import sys
import numpy as np
import cv2
import argparse
import math
import random
import glob
import time
import json
from itertools import product
from skimage.filters import threshold_sauvola, threshold_otsu

# global definitions
SEED = 0
random.seed(SEED)

# https://code.google.com/archive/p/isri-ocr-evaluation-tools
ISRI_DATASET_DIR = 'datasets/isri-ocr'
CDIP_DATASET_DIR = 'datasets/cdip'


def create_directory(dir_):
    ''' Create a directory if it does no exist. '''

    if not os.path.exists(dir_):
        os.makedirs(dir_)


def init(args):
    ''' Initial setup. '''

    # build tree directory
    for class_ in ['positives', 'negatives']:
        create_directory('{}/{}/train'.format(args.save_dir, class_))
        create_directory('{}/{}/val'.format(args.save_dir, class_))


def is_neutral(sample, thresh):
    ''' Check whether a sample is neutral. '''

    return ((~sample).sum() / sample.size) < thresh


def apply_noise(sample, half, radius):
    ''' Apply noise onto a sample. '''

    noisy = sample.copy()
    noise = np.random.choice([False, True], (sample.shape[0], 2 * radius))
    noisy[:, half - radius : half + radius] |= noise
    return noisy


def save_png(path, image):
    ''' Save image in png format. '''

    image = (255 * image).astype(np.uint8)
    cv2.imwrite('{}.png'.format(path), image)


def save_npy(path, image):
    ''' Save image in npy format. '''

    cv2.imwrite('{}.npy'.format(path), image)


def generate_samples(args, sample_size=(32, 32)):
    ''' Sampling process. '''

    if glob.glob('{}/**/*.{}'.format(args.save_dir, args.extension), recursive=True):
        print('Sampling already done!')
        sys.exit()

    # defines function to save image
    save_fn = save_png if args.extension == 'png' else save_npy

    # dataset
    docs = []
    if args.dataset == 'cdip':
        docs = glob.glob('{}/ORIG_*/*.tif'.format(CDIP_DATASET_DIR))
    else:
        docs = glob.glob('{}/**/*.tif'.format(ISRI_DATASET_DIR), recursive=True)
    random.shuffle(docs)
    if args.num_docs is not None:
        docs = docs[ : args.num_docs]

    # split train and validation sets
    num_docs = len(docs)
    docs_train = docs[int(args.pval * num_docs) :]
    docs_val = docs[ : int(args.pval * num_docs)]

    processed = 0
    thresh_func = threshold_sauvola if args.thresh_method == 'sauvola' else threshold_otsu
    size_left = sample_size[1] // 2
    size_right = sample_size[1] - size_left
    for mode, docs in zip(['train', 'val'], [docs_train, docs_val]):
        count = {'positives': 0, 'negatives': 0}

        for doc in docs:
            processed += 1

            # load
            image = cv2.imread(doc, cv2.IMREAD_GRAYSCALE)
            height, width = image.shape

            # threshold
            thresh = thresh_func(image)
            # thresholded = (255 * (image > thresh)).astype(np.uint8)
            thresholded = (image > thresh)

            # shredding
            print('     => Shredding')
            count_doc = {'positives': 0, 'negatives': 0}
            acc = 0
            strips = []
            for i in range(args.num_strips):
                dw = int((width - acc) / (args.num_strips - i))
                strip = thresholded[:, acc : acc + dw]
                strips.append(strip)
                acc += dw

            # sampling
            print('     => Sampling')
            N = len(strips)
            pos_combs = [(i, i + 1) for i in range(N - 1)]
            neg_combs = [(i, j) for i in range(N) for j in range(N) if (i != j) and (i + 1 != j)]
            random.shuffle(pos_combs)
            random.shuffle(neg_combs)

            # alternate sampling (pos1 -> neg1, pos2 -> neg2, ...)
            for pos_comb, neg_comb in zip(pos_combs, neg_combs):
                for (i, j), label in zip([pos_comb, neg_comb], ['positives', 'negatives']):
                    print('document {}/{}[mode={}] :: '.format(processed, num_docs, mode), end='')
                    print('({},{}) :: positives={} negatives={}'.format(
                        i, j, count['positives'], count['negatives'])
                    )

                    # sampling
                    stop = False
                    image = np.hstack([strips[i][:, -size_left :], strips[j][:, : size_right]])
                    for y in range(0, height - sample_size[0], args.stride):
                        sample = image[y : y + sample_size[0]]

                        # not a neutral sample and positives/negatives samples limit per document was not reached?
                        if not is_neutral(sample, args.neutral_thresh) and count_doc[label] < args.max_samples:
                            # apply noise and save
                            noisy = apply_noise(sample, size_left, args.noise_radius)
                            save_fn('{}/{}/{}/{}'.format(args.save_dir, label, mode, count[label]), noisy)
                            count_doc[label] += 1
                            count[label] += 1

                # overall limit was reached?
                if sum(count_doc.values()) == (3 * args.max_samples):
                    break


def generate_files(args):
    ''' Generate train.txt and val.txt. '''

    docs_neg_train = glob.glob('{}/negatives/train/*.{}'.format(args.save_dir, args.extension))
    docs_neg_val = glob.glob('{}/negatives/val/*.{}'.format(args.save_dir, args.extension))
    docs_pos_train = glob.glob('{}/positives/train/*.{}'.format(args.save_dir, args.extension))
    docs_pos_val = glob.glob('{}/positives/val/*.{}'.format(args.save_dir, args.extension))

    # labels
    #   0: negative
    #   1: positive
    neg_train = ['{} 0'.format(doc) for doc in docs_neg_train]
    pos_train = ['{} 1'.format(doc) for doc in docs_pos_train]
    neg_val = ['{} 0'.format(doc) for doc in docs_neg_val]
    pos_val = ['{} 1'.format(doc) for doc in docs_pos_val]

    train = neg_train + pos_train
    val = neg_val + pos_val
    random.shuffle(train)
    random.shuffle(val)

    open('{}/train.txt'.format(args.save_dir), 'w').write('\n'.join(train))
    open('{}/val.txt'.format(args.save_dir), 'w').write('\n'.join(val))

    # general info
    stats = {
        'negatives_train': len(docs_neg_train),
        'positives_train': len(docs_pos_train),
        'negatives_val': len(docs_neg_val),
        'positives_val': len(docs_pos_val)
    }
    info = {
        'stats': stats,
        'params': args.__dict__,
    }
    json.dump(info, open('{}/info.json'.format(args.save_dir), 'w'))


if __name__ == '__main__':
    t0 = time.time()

    parser = argparse.ArgumentParser(description='Dataset samples generation.')
    parser.add_argument(
        '-v', '--pval', action='store', dest='pval', required=False, type=float,
        default=0.1, help='Percentage of samples reserved for validation.'
    )
    parser.add_argument(
        '-ms', '--max-samples', action='store', dest='max_samples', required=False, type=int,
        default=1000, help='Maximum number of samples of a given type (positive, negative, neutral) per document.'
    )
    parser.add_argument(
        '-nt', '--neutral-thresh', action='store', dest='neutral_thresh', required=False, type=float,
        default=0.2, help='Neutral sample threshold.'
    )
    parser.add_argument(
        '-nr', '--noise-radius', action='store', dest='noise_radius', required=False, type=int,
        default=2, help='Noise radius.'
    )
    parser.add_argument(
        '-nd', '--num-docs', action='store', dest='num_docs', required=False, type=int,
        default=None, help='Number of documents.'
    )
    parser.add_argument(
        '-d', '--dataset', action='store', dest='dataset', required=False, type=str,
        default='cdip', help='Dataset.'
    )
    parser.add_argument(
        '-ss', '--sample-size', action='store', dest='sample_size', required=False, nargs=2, type=int,
        default=[32, 32], help='Sample size (H x W).'
    )
    parser.add_argument(
        '-ns', '--num-strips', action='store', dest='num_strips', required=False, type=int,
        default=30, help='Number of strips (simulated shredding).'
    )
    parser.add_argument(
        '-sd', '--save-dir', action='store', dest='save_dir', required=False, type=str,
        default='datasets/samples', help='Path where samples will be placed.'
    )
    parser.add_argument(
        '-tm', '--thresh-method', action='store', dest='thresh_method', required=False, type=str,
        default='sauvola', help='Thresholding method.'
    )
    parser.add_argument(
        '-st', '--stride', action='store', dest='stride', required=False, type=int,
        default=2, help='Stride.'
    )
    parser.add_argument(
        '-if', '--extension', action='store', dest='extension', required=False, type=str,
        default='png', help='Image format (extension).'
    )
    args = parser.parse_args()
    sample_size = tuple(args.sample_size)

    assert args.dataset in ['cdip', 'isri-ocr']
    assert args.thresh_method in ['otsu', 'sauvola']
    assert args.extension in ['png', 'npy']
    # assert sample_size in list(product([32, 48, 64], [32, 48, 64]))

    init(args)
    print('Extracting characters')
    generate_samples(args, sample_size=sample_size)
    print('Writing files')
    generate_files(args)

    t1 = time.time()
    print('Elapsed time={:.2f} minutes ({} seconds)'.format((t1 - t0) / 60.0, t1 - t0))

    info = json.load(open('{}/info.json'.format(args.save_dir), 'r'))
    info['time_minutes'] = (t1 - t0) / 60.0
    info['time_seconds'] = (t1 - t0)
    json.dump(info, open('{}/info.json'.format(args.save_dir), 'w'))