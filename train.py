import os
import argparse
import sys
import json
import math
import time

import numpy as np
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.utils import shuffle
from sklearn.metrics import silhouette_score
import multiprocessing as mp

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from docrec.models.affinet import AffiNET

CPUS_RATIO = 0.5 # % of cpus used for dataset processing
SEED = 0 # <= change this in case of multiple runs
np.random.seed(SEED)
tf.set_random_seed(SEED)


def cohen_d(x, y):
    nx = x.size
    ny = y.size
    dof = nx + ny - 2
    return (x.mean() - y.mean()) / np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1 ) * y.std(ddof=1) ** 2) / dof)


class Dataset:

    def __init__(self, args, mode='train', sess=None):

        assert mode in ['train', 'val']

        lines = open('{}/{}.txt'.format(args.samples_dir, mode)).readlines()
        info = json.load(open('{}/info.json'.format(args.samples_dir), 'r'))
        num_negatives = info['stats']['negatives_{}'.format(mode)]
        num_positives = info['stats']['positives_{}'.format(mode)]
        num_samples_per_class = min(num_positives, num_negatives)

        self.num_samples = 2 * num_samples_per_class
        self.curr_epoch = 1
        self.num_epochs = args.num_epochs
        self.curr_batch = 1
        self.batch_size = args.batch_size
        self.sample_size = info['params']['sample_size']
        self.num_batches = math.ceil(self.num_samples / self.batch_size)
        self.sess = sess

        assert self.num_samples > self.batch_size

        def _parse_function(filename, label):
            ''' Parse function. '''

            image_string = tf.read_file(filename)
            image = tf.image.decode_jpeg(image_string, channels=3, dct_method='INTEGER_ACCURATE') # works with png too
            image = tf.image.convert_image_dtype(image, tf.float32)
            image_left, image_right = tf.split(image, 2, axis=1) # split horizontally images into halves
            # if mode == 'train':
            #     return image_left, image_right, tf.one_hot(label, NUM_CLASSES)
            return image_left, image_right, label


        # load samples' filenames and labels
        count = {'0': 0, '1': 0} # balancing classes
        labels = []
        filenames = []
        for line in lines:
            filename, label = line.split()
            if count[label] < num_samples_per_class:
                filenames.append(filename)
                labels.append(int(label))
                count[label] += 1

        # dataset iterator
        dataset_tf = tf.data.Dataset.from_tensor_slices((filenames, labels))
        if mode == 'train':
            dataset_tf = dataset_tf.shuffle(len(filenames), seed=SEED) # important => reshuffle each iteraction

        dataset_tf = dataset_tf.map(_parse_function, num_parallel_calls=max(1, int(CPUS_RATIO * mp.cpu_count())))
        dataset_tf = dataset_tf.batch(args.batch_size).repeat(args.num_epochs)
        self.next_batch_op = dataset_tf.make_one_shot_iterator().get_next()


    def next_batch(self):

        return self.sess.run(self.next_batch_op)


def train(args):

    # reset the default graph and init a session
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # create traindata directory if not existing (this directory is not syncronized in github)
    os.makedirs('traindata', exist_ok=True)

    # load samples dataset / samples size
    print('loading training samples :: ', end='')
    sys.stdout.flush()
    dataset = Dataset(args, mode='train', sess=sess)
    sample_height, sample_width = dataset.sample_size
    print('num_samples={} sample_size={}x{}'.format(dataset.num_samples, sample_height, sample_width))

    # placeholders
    # 1) images (channels last)
    images_left_ph = tf.placeholder(
        tf.float32, name='left_ph', shape=(None, sample_height, sample_width // 2, 3)
    )
    images_right_ph = tf.placeholder(
        tf.float32, name='right_ph', shape=(None, sample_height, sample_width - sample_width // 2, 3)
    )
    # 2) labels
    # compatible 1 (+)
    # uncompatible 0 (-)
    labels_ph = tf.placeholder(tf.float32, name='labels_ph', shape=(None,))

    # models
    net_left = AffiNET(
        images_left_ph, feat_dim=args.feat_dim, feat_layer=args.feat_layer, mode='train',
        pretrained=False, base_arch='squeezenet', channels_first=False, activation=args.activation,
        sample_height=sample_height, model_scope='left', seed=SEED, sess=sess
    )
    net_right = AffiNET(
        images_right_ph, feat_dim=args.feat_dim, feat_layer=args.feat_layer, mode='train',
        pretrained=False, base_arch='squeezenet', channels_first=False, activation=args.activation,
        sample_height=sample_height, model_scope='right', seed=SEED, sess=sess
    )

    # features: batch x n_features dimensions, where n_features = height * dim_feat
    pred_left = tf.squeeze(net_left.features, 1)
    pred_right = tf.squeeze(net_right.features, 1)

    # contrastive loss function
    # http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    # https://stackoverflow.com/questions/41172500/how-to-implement-metrics-learning-using-siamese-neural-network-in-tensorflow
    d = tf.reduce_sum(tf.square(tf.subtract(pred_left, pred_right)), 1) # sum feature-wise
    d_sqrt = tf.sqrt(d)
    loss_op = (1.0 - labels_ph) * tf.square(tf.maximum(0.0, args.margin - d_sqrt)) + labels_ph * d
    loss_op = 0.5 * tf.reduce_mean(loss_op) # averaging on batch

    # optimizer (SGD)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
    train_op = optimizer.minimize(loss_op)
    # grads_and_vars = optimizer.compute_gradients(loss_op)
    # # train_op = optimizer.apply_gradients(grads_and_vars)

    # init variables
    sess.run(tf.global_variables_initializer())

    # training loop
    start = time.time()
    losses_group = []
    losses = []
    steps = []
    global_step = 0
    num_steps_per_epoch = math.ceil(dataset.num_samples / args.batch_size)
    total_steps = args.num_epochs * num_steps_per_epoch
    for epoch in range(1, args.num_epochs + 1):
        for step in range(1, num_steps_per_epoch + 1):
            # batch data
            images_left, images_right, labels = dataset.next_batch()
            # train
            loss, _, dists = sess.run(
                [loss_op, train_op, d_sqrt],
                feed_dict={images_left_ph: images_left, images_right_ph: images_right, labels_ph: labels}
            )
            losses_group.append(loss)
            if (step % 100 == 0) or (step == num_steps_per_epoch):
                losses.append(np.mean(losses_group))
                steps.append(global_step)
                elapsed = time.time() - start
                remaining = elapsed * (total_steps - global_step) / global_step
                print('[{:.2f}%] step={}/{} epoch={} loss={:.3f} :: {:.2f}/{:.2f} seconds lr={}'.format(
                    100 * global_step / total_steps, global_step, total_steps, epoch,
                    np.mean(losses_group), elapsed, remaining, args.learning_rate
                ))
                losses_group = []
                labels_arr = np.array(labels.tolist())
                dists_arr = np.array(dists.tolist())
                print('dist(-)={:.4f} dist(+)={:.4f}'.format(
                    dists_arr[labels_arr == 0].mean(), dists_arr[labels_arr == 1].mean())
                )

            # increment global step
            global_step += 1

        # save epoch model
        net_left.save_weights('traindata/{}/model/left/{}.npy'.format(args.model_id, epoch))
        net_right.save_weights('traindata/{}/model/right/{}.npy'.format(args.model_id, epoch))

    plt.plot(steps, losses)
    plt.savefig('traindata/{}/loss.png'.format(args.model_id))
    sess.close()


def validate(args):

    # reset the default graph and init a session
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # load samples dataset / samples size
    print('loading validation samples :: ', end='')
    sys.stdout.flush()
    dataset = Dataset(args, mode='val', sess=sess)
    sample_height, sample_width = dataset.sample_size
    print('num_samples={} sample_size={}x{}'.format(dataset.num_samples, sample_height, sample_width))

    # placeholders
    # 1) images
    images_left_ph = tf.placeholder(
        tf.float32, name='left_ph', shape=(None, sample_height, sample_width // 2, 3)
    )   # channels last
    images_right_ph = tf.placeholder(
        tf.float32, name='right_ph', shape=(None, sample_height, sample_width - sample_width // 2, 3)
    ) # channels last

    # models
    net_left = AffiNET(
        images_left_ph, feat_dim=args.feat_dim, feat_layer=args.feat_layer, mode='val',
        base_arch='squeezenet', channels_first=False, activation=args.activation,
        sample_height=sample_height, model_scope='left', sess=sess
    )
    net_right = AffiNET(
        images_right_ph, feat_dim=args.feat_dim, feat_layer=args.feat_layer, mode='val',
        base_arch='squeezenet', channels_first=False, activation=args.activation,
        sample_height=sample_height, model_scope='right', sess=sess
    )

    # features: batch x n_features dimensions
    pred_left = tf.squeeze(net_left.features, 1)   # batch x (height * dim_feat)
    pred_right = tf.squeeze(net_right.features, 1) # batch x (height * dim_feat)

    # distances
    dists_op = tf.reduce_sum(tf.square(tf.subtract(pred_left, pred_right)), 1) # sum feature-wise

    # init variables
    sess.run(tf.global_variables_initializer())

    # validation loop
    max_accuracy = -1
    best_epoch = -1
    accuracies = []
    num_steps_per_epoch = math.ceil(dataset.num_samples / args.batch_size)
    total_steps = args.num_epochs * num_steps_per_epoch
    for epoch in range(1, args.num_epochs + 1):
        net_left.load_weights('traindata/{}/model/left/{}.npy'.format(args.model_id, epoch))
        net_right.load_weights('traindata/{}/model/right/{}.npy'.format(args.model_id, epoch))

        dists = []
        labels = []
        for step in range(1, num_steps_per_epoch + 1):
            images_left, images_right, labels_step = dataset.next_batch()
            dists_step = sess.run(
                dists_op, feed_dict={images_left_ph: images_left, images_right_ph: images_right}
            )
            dists.append(dists_step)
            labels.append(labels_step)

        # calculate accuracy
        dists_arr = np.concatenate(dists)
        labels_arr = np.concatenate(labels)
        accuracy = cohen_d(dists_arr[labels_arr == 0], dists_arr[labels_arr == 1])
        accuracies.append(accuracy)
        if accuracy > max_accuracy: # grab the highest / best epoch
            max_accuracy = accuracy
            best_epoch = epoch

        print('[{:.2f}%] epoch={}/{} (best={}) accuracy={:.5f} (max={:.5f})'.format(
            100 * epoch / args.num_epochs, epoch, args.num_epochs, best_epoch,
            accuracy, max_accuracy
        ))


    plt.cla()
    plt.plot(np.arange(1, args.num_epochs + 1), accuracies)
    # plt.xticks(np.arange(1, args.num_epochs + 1))
    plt.vlines(best_epoch, 0.0, 1.0, colors='r', linestyles='dashed')
    plt.savefig('traindata/{}/accuracy.png'.format(args.model_id))
    sess.close()

    return best_epoch


def main():

    parser = argparse.ArgumentParser(description='Training the networks.')
    parser.add_argument(
        '-lr', '--learning-rate', action='store', dest='learning_rate', required=False, type=float,
        default=0.1, help='Learning rate.'
    )
    parser.add_argument(
        '-bs', '--batch-size', action='store', dest='batch_size', required=False, type=int,
        default=256, help='Batch size.'
    )
    parser.add_argument(
        '-e', '--epochs', action='store', dest='num_epochs', required=False, type=int,
        default=30, help='Number of training epochs.'
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
        '-t', '--top', action='store', dest='top', required=False, type=int,
        default=10, help='Top <x> for validation.'
    )
    parser.add_argument(
        '-s', '--step-size', action='store', dest='step_size', required=False, type=float,
        default=0.33, help='Step size for learning with step-down policy.'
    )
    parser.add_argument(
        '-a', '--activation', action='store', dest='activation', required=False, type=str,
        default='sigmoid', help='Activation function (final net layer).'
    )
    parser.add_argument(
        '-sd', '--samples-dir', action='store', dest='samples_dir', required=False, type=str,
        default='~/datasets/samples', help='Path where samples (samples) are placed.'
    )
    parser.add_argument(
        '-d', '--dataset', action='store', dest='dataset', required=False, type=str,
        default='cdip', help='Dataset id.'
    )
    parser.add_argument(
        '-ma', '--margin', action='store', dest='margin', required=False, type=float,
        default=1.0, help='Margin for contrastive divergence loss'
    )
    parser.add_argument(
        '-m', '--model-id', action='store', dest='model_id', required=False, type=str,
        default=None, help='Model identifier (tag).'
    )
    args = parser.parse_args()

    # training stage
    t0 = time.time()
    train(args)
    train_time = time.time() - t0

    # validation
    t0 = time.time()
    best_epoch = validate(args)
    val_time = time.time() - t0

    # dump training info
    info = {
        'train_time': train_time,
        'val_time': val_time,
        'sample_height': int(args.samples_dir.split('_')[-1].split('x')[0]),
        'best_model_left': 'traindata/{}/model/left/{}.npy'.format(args.model_id, best_epoch),
        'best_model_right': 'traindata/{}/model/right/{}.npy'.format(args.model_id, best_epoch),
        'params': args.__dict__
    }
    json.dump(info, open('traindata/{}/info.json'.format(args.model_id), 'w'))
    return train_time + val_time


if __name__ == '__main__':

    t = main()
    print('Elapsed time={:.2f} minutes ({} seconds)'.format(t / 60.0, t))
