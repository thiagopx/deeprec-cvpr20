from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from time import time
import cv2
from skimage.filters import threshold_sauvola, threshold_otsu
import numpy as np
import math
import tensorflow as tf
# from keras import backend as K

from .algorithm import Algorithm
from ..neural.models.affinet import AffiNET


class Proposed(Algorithm):
    '''  Proposed algorithm. '''

    def __init__(
        self, weights_path_left, weights_path_right, vshift, input_size, feat_dim=64,
        feat_layer='drop9', activation='sigmoid', sample_height=32, thresh_method='sauvola', verbose=False
    ):

        assert thresh_method in ['otsu', 'sauvola']

        # preparing models
        h, w = input_size
        self.wl = int(w / 2)
        self.wr = w - self.wl
        self.vshift = vshift
        self.input_size_h, self.input_size_w = input_size
        self.images_left_ph = tf.placeholder(
            tf.float32, name='images_left_ph', shape=(1, self.input_size_h, self.wl, 3) # channels last
        )
        self.images_right_ph = tf.placeholder(
            tf.float32, name='images_right_ph', shape=(1, self.input_size_h, self.wr, 3) # channels last
        )

        # models
        net_left = AffiNET(
            self.images_left_ph, feat_dim=feat_dim, feat_layer=feat_layer, mode='test',
            base_arch='squeezenet', channels_first=False, activation=activation,
            sample_height=sample_height, model_scope='left'
        )
        net_right = AffiNET(
            self.images_right_ph, feat_dim=feat_dim, feat_layer=feat_layer, mode='test',
            base_arch='squeezenet', channels_first=False, activation=activation,
            sample_height=sample_height, model_scope='right',
            sess=net_left.sess
        )

        # features: height x n_features (squeeze batch dimension)
        self.pred_left = tf.squeeze(net_left.features, 0)
        self.pred_right = tf.squeeze(net_right.features, 0)

        # result
        self.compatibilities = None
        self.displacements = None

        # init model
        self.sess = net_left.sess
        self.sess.run(tf.global_variables_initializer())
        net_left.load_weights(weights_path_left)
        net_right.load_weights(weights_path_right)

        self.verbose = verbose
        self.inferente_time = 0
        self.thresh_method = thresh_method


    def _extract_features(self, strip):
        ''' Extract image around the border. '''

        image = cv2.cvtColor(strip.filled_image(), cv2.COLOR_RGB2GRAY)
        thresh_func = threshold_sauvola if self.thresh_method == 'sauvola' else threshold_otsu
        thresh = thresh_func(image)
        thresholded = (image > thresh).astype(np.float32)

        image_bin = np.stack(3 * [thresholded]).transpose((1, 2, 0)) # channels last

        wl = self.wl
        wr = self.wr
        h, w, _ = strip.image.shape
        offset = int((h - self.input_size_h) / 2)

        # features of the left side
        left_border = strip.offsets_l
        image = np.ones((self.input_size_h, wl, 3), dtype=np.float32)
        for y, x in enumerate(left_border[offset : offset + self.input_size_h]):
            w_new = min(wl, w - x)
            image[y, : w_new] = image_bin[y + offset, x : x + w_new]

        # use the right network for the left border
        left = self.sess.run(self.pred_right, feed_dict={self.images_right_ph: image[np.newaxis]})

        # features of the right side
        right_border = strip.offsets_r
        image = np.ones((self.input_size_h, wr, 3), dtype=np.float32)
        for y, x in enumerate(right_border[offset : offset + self.input_size_h]):
            w_new = min(wr, x + 1)
            image[y, : w_new] = image_bin[y + offset, x - w_new + 1: x + 1]

        # use the left network for the right border
        right = self.sess.run(self.pred_left, feed_dict={self.images_left_ph: image[np.newaxis]})

        return left, right


    def run(self, strips, d=0): # d is not used at this moment
        ''' Run algorithm. '''

        N = len(strips.strips)
        compatibilities = np.zeros((N, N), dtype=np.float32)
        displacements = np.zeros((N, N), dtype=np.int32)
        wr = self.wr

        if self.verbose:
            print('extracting features ', end='')
            t0 = time()

        # features
        features = []
        for strip in strips.strips:
            left, right = self._extract_features(strip)
            features.append((left, right))

        if self.verbose:
            print(':: elapsed={:.2f}s'.format(time() - t0))

        nrows = features[0][0].shape[0] - self.vshift
        inference = 1
        self.inference_time = 0
        total_inferences = N * (N - 1)
        for i in range(N): # for each strip
            feat_i = features[i][1] # right border (left strip)

            for j in range(N):
                if i == j:
                    continue

                best_disp = -self.vshift
                feat_j = features[j][0] # left border (right strip)
                t0 = time()
                min_dist = np.sum((feat_j[-best_disp : nrows - best_disp] - feat_i[ : nrows]) ** 2)
                t1 = time()
                for disp in range(-self.vshift + 1, self.vshift + 1):
                    i1 = max(0, disp)
                    i2 = i1 + nrows
                    j1 = max(0, -disp)
                    j2 = j1 + nrows

                    t2 = time()
                    dist = np.sum((feat_j[j1 : j2] - feat_i[i1 : i2]) ** 2)
                    t3 = time()

                    if dist < min_dist:
                        min_dist = dist
                        best_disp = disp

                compatibilities[i, j] = min_dist
                displacements[i, j] = best_disp
                self.inference_time += (t1 - t0) + (t3 - t2)

                if self.verbose and (inference % 50 == 0):
                    remaining = self.inference_time * (total_inferences - inference) / inference
                    print('[{:.2f}%] inference={}/{} :: elapsed={:.2f}s predicted={:.2f}s mean inf. time={:.10f}s '.format(
                        100 * inference / total_inferences, inference, total_inferences, self.inference_time,
                        remaining, self.inference_time / inference
                    ))
                inference += 1

        self.compatibilities = compatibilities
        self.displacements = displacements
        return self


    def name(self):
        ''' Method name. '''

        return 'proposed'
