
import os
import numpy as np
import tensorflow as tf


# https://github.com/vonclites/squeezenet/blob/master/networks/squeezenet.py
# https://github.com/ethereon/caffe-tensorflow
# python convert.py --caffemodel <path-caffemodel> <path-deploy_prototxt> --data-output-path <path_weights> <path_model>py>

class AffiNET:

    def __init__(
        self, input_tensor, feat_dim=64, mode='train', model_scope='AffiNET', activation='relu',
        sample_height=32, channels_first=False, seed=None, sess=None
    ):
        ''' Affinet: Learning Compatibility between Shreds for Document Reconstruction. '''

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

        data_format = 'channels_first' if channels_first else 'channels_last'
        concat_axis = 1 if channels_first else 3
        height_input = input_tensor.get_shape().as_list()[1]
        with tf.variable_scope(model_scope, reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(
                input_tensor, 64, 3, 2, padding='valid', activation=tf.nn.relu, data_format=data_format, name='conv1',
                kernel_initializer=tf.glorot_uniform_initializer(seed=seed)
            )
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=2, data_format=data_format)
            fire2_squeeze1x1 = tf.layers.conv2d(
                pool1, 16, 1, activation=tf.nn.relu, data_format=data_format, name='fire2_squeeze1x1',
                kernel_initializer=tf.glorot_uniform_initializer(seed=seed)
            )
            fire2_expand1x1 = tf.layers.conv2d(
                fire2_squeeze1x1, 64, 1, activation=tf.nn.relu, data_format=data_format, name='fire2_expand1x1',
                kernel_initializer=tf.glorot_uniform_initializer(seed=seed)
            )
            fire2_expand3x3 = tf.layers.conv2d(
                fire2_squeeze1x1, 64, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire2_expand3x3',
                kernel_initializer=tf.glorot_uniform_initializer(seed=seed)
            )
            fire2_concat = tf.concat([fire2_expand1x1, fire2_expand3x3], axis=concat_axis)

            # features
            last = fire2_concat
            _, height, width, depth = last.get_shape().as_list()
            height_reduction_factor = height / height_input
            sample_height_reduced = int(sample_height * height_reduction_factor)

            print(_, height, sample_height_reduced, width, depth, '(1)')
            features = tf.layers.conv2d(
                last, feat_dim, (sample_height_reduced, width), kernel_initializer=tf.random_normal_initializer(0.0, 0.01, seed=seed),
                activation=activation_fn, data_format=data_format, name='features'
            )
            _, height, width, depth = features.get_shape()
            print(_, height, width, depth, '(2)')

            # the model is designed to be batch x height (strided) x 1 x dim_feat
            assert width == 1
            # flatten (squeeze width)
            self.features = tf.reshape(features, [-1, height, depth]) # batch x height (strided) x dim_feat


    def load_weights(self, weights_path, ignore_layers=[], BGR=False, ignore_missing=False):
        ''' Load network weights and biases (format caffe-tensorflow).
        weights_path: path to the numpy-serialized network weights.
        ignore_layers: layers whose parameters must be ignored.
        BGR: if data is BGR, convert weights from the first layer to RGB.
        ignore_missing: if true, serialized weights for missing layers are ignored.
        '''

        first_layer='conv1'
        data_dict = np.load(weights_path, encoding='latin1').item()
        for layer in data_dict:
            if layer in ignore_layers:
                continue
            for param_name, data in data_dict[layer].items():
                param_name = param_name.replace('weights', 'kernel').replace('biases', 'bias')
                try:
                    scope = '{}/{}'.format(self.scope, layer) if self.scope else layer
                    with tf.variable_scope(scope, reuse=True):
                        var = tf.get_variable(param_name)
                        if (layer == first_layer) and BGR and (param_name == 'kernel'):
                            data = data[:, :, [2, 1, 0], :] # BGR => RGB
                        self.sess.run(var.assign(data))
                except ValueError:
                    if not ignore_missing:
                        raise


    def save_weights(self, weights_path, ignore_layers=[]):
        ''' Load network weights and biases (format caffe-tensorflow).
        data_path: path to the numpy-serialized network weights.
        session: current TensorFlow session.
        ignore_layers: layers whose parameters must be ignored.
        '''

        data_dict = {}
        for var in tf.trainable_variables():
            layer, param_name = var.op.name.split('/')[-2 :] # excluce scope if existing
            if layer in ignore_layers:
                continue
            data = self.sess.run(var)
            try:
                data_dict[layer][param_name] = data
            except KeyError:
                data_dict[layer] = {param_name: data}

        # ckeck directory path
        if not os.path.exists(os.path.dirname(weights_path)):
            os.makedirs(os.path.dirname(weights_path))
        np.save(weights_path, np.array(data_dict))
