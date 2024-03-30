
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from .squeezenet import SqueezeNet


class AffiNET:

    def __init__(
        self, input_tensor, feat_dim=64, feat_layer='fire9', mode='train',
        base_arch='squeezenet', pretrained=False, model_scope='AffiNET',
        activation='relu', sample_height=32, channels_first=False, seed=None, sess=None
    ):
        ''' Affinet: Learning Compatibility between Shreds for Document Reconstruction. '''

        assert base_arch in ['squeezenet']
        assert mode in ['train', 'val', 'test']
        assert activation in ['relu', 'sigmoid', 'linear']

        activation_fn = None # linear activation
        if activation == 'relu':
            activation_fn = tf.nn.relu
        elif activation == 'sigmoid':
            activation_fn = tf.nn.sigmoid

        self.sess = sess
        if self.sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

        self.scope = model_scope
        self.base_model = None

        data_format = 'channels_first' if channels_first else 'channels_last'
        concat_axis = 1 if channels_first else 3
        height_input = input_tensor.get_shape().as_list()[1]
        with tf.variable_scope(model_scope, reuse=tf.AUTO_REUSE):
            if base_arch == 'squeezenet':
                self.base_model = SqueezeNet(
                    input_tensor, include_top=False, feat_layer=feat_layer, mode=mode,
                    model_scope='SqueezeNet', channels_first=False, sess=self.sess, seed=seed
                )
                # imagenet pretrain
                if mode == 'train' and pretrained:
                    self.base_model.load_weights(
                        'docrec/neural/models/imagenet.npy', ignore_layers=['conv10'],
                        BGR=True, ignore_missing=True
                    )
            # features
            input_layer = self.base_model.output

            _, height, width, depth = input_layer.get_shape().as_list()
            height_reduction_factor = height / height_input
            sample_height_reduced = int(sample_height * height_reduction_factor) # sample


            # slidind window with visual field defined by the sample size
            features = tf.layers.conv2d(
                input_layer, feat_dim, (sample_height_reduced, width),
                kernel_initializer=tf.random_normal_initializer(0.0, 0.01, seed=seed),
                activation=activation_fn, data_format=data_format, name='features'
            )
            _, height, width, depth = features.get_shape()

            # the model is designed to be batch x height (strided) x 1 x dim_feat
            assert width == 1
            # remove width dimension
            self.features = tf.reshape(features, [-1, height, depth]) # batch x height (strided) x dim_feat


    def load_weights(self, weights_path, ignore_layers=[], ignore_missing=False):
        ''' Load network weights and biases (format caffe-tensorflow).
        weights_path: path to the numpy-serialized network weights.
        ignore_layers: layers whose parameters must be ignored.
        BGR: if data is BGR, convert weights from the first layer to RGB.
        ignore_missing: if true, serialized weights for missing layers are ignored.
        '''

        data_dict = np.load(weights_path, encoding='latin1', allow_pickle=True).item()
        for var_name, data in data_dict.items():
            with tf.variable_scope(self.scope, reuse=True):
                var = tf.get_variable(var_name)
                self.sess.run(var.assign(data))


    def save_weights(self, weights_path):
        ''' Load network weights and biases (format caffe-tensorflow).
        weigths_path: path to the numpy-serialized network weights.
        '''

        data_dict = {}
        vars = tf.trainable_variables(scope=self.scope)
        for var in vars:
            var_name = var.op.name.replace('{}/'.format(self.scope), '')
            data_dict[var_name] = self.sess.run(var)

        # ckeck directory path
        if not os.path.exists(os.path.dirname(weights_path)):
            os.makedirs(os.path.dirname(weights_path))
        np.save(weights_path, np.array(data_dict))
