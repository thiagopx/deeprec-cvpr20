import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from collections import OrderedDict


# https://github.com/vonclites/squeezenet/blob/master/networks/squeezenet.py
# https://github.com/ethereon/caffe-tensorflow
# python convert.py --caffemodel <path-caffemodel> <path-deploy_prototxt> --data-output-path <path_weights> <path_model>py>

def fire_module(input_layer, s_1x1, e_1x1, e_3x3, fire_id, data_format='channels_last'):

    concat_axis = 1 if data_format=='channels_first' else 3
    squeeze1x1 = tf.layers.conv2d(
        input_layer, s_1x1, 1, activation=tf.nn.relu, data_format=data_format,
        name='fire{}_squeeze1x1'.format(fire_id)
    )
    expand1x1 = tf.layers.conv2d(
        squeeze1x1, e_1x1, 1, activation=tf.nn.relu, data_format=data_format,
        name='fire{}_expand1x1'.format(fire_id)
    )
    expand3x3 = tf.layers.conv2d(
        squeeze1x1, e_3x3, 3, padding='same', activation=tf.nn.relu, data_format=data_format,
        name='fire{}_expand3x3'.format(fire_id)
    )
    concat = tf.concat([expand1x1, expand3x3], axis=concat_axis)
    return concat


class SqueezeNet:

    def __init__(
        self, input_tensor, include_top=True, feat_layer='drop9', num_classes=1000,
        mode='train', model_scope='SqueezeNet', channels_first=False, sess=None, seed=0
    ):
        ''' SqueezeNet v1.1
        Adpated from the Caffe original implementation: https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1
        Reference:
        @article{iandola2016squeezenet,
            title={Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size},
            author={Iandola, Forrest N and Han, Song and Moskewicz, Matthew W and Ashraf, Khalid and Dally, William J and Keutzer, Kurt},
            journal={arXiv preprint arXiv:1602.07360},
            year={2016}
        }
        '''

        assert mode in ['train', 'val', 'test']

        self.sess = sess
        if self.sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

        self.scope = model_scope

        # structure definition
        layers_params = OrderedDict()
        layers_params['pool1'] = [3, 2]
        layers_params['fire2'] = [16, 64, 64, 2]
        layers_params['fire3'] = [16, 64, 64, 3]
        layers_params['pool3'] = [3, 2]
        layers_params['fire4'] = [32, 128, 128, 4]
        layers_params['fire5'] = [32, 128, 128, 5]
        layers_params['pool5'] = [3, 2]
        layers_params['fire6'] = [48, 192, 192, 6]
        layers_params['fire7'] = [48, 192, 192, 7]
        layers_params['fire8'] = [64, 256, 256, 8]
        layers_params['fire9'] = [64, 256, 256, 9]
        layers_params['drop9'] = [0.5]
        assert feat_layer in layers_params

        # network building
        # layers = {}
        data_format = 'channels_first' if channels_first else 'channels_last'
        concat_axis = 1 if channels_first else 3
        with tf.variable_scope(model_scope, reuse=tf.AUTO_REUSE):
            current = tf.layers.conv2d(
                input_tensor, 64, 3, 2, padding='valid', activation=tf.nn.relu,
                data_format=data_format, name='conv1'
            )

            # appending intermediate layers
            # current = layers['conv1']
            for layer_id, params in layers_params.items():
                # create layer according its type
                # print(layer_id, params)
                if layer_id.startswith('fire'):
                    current = fire_module(current, params[0], params[1], params[2], params[3], data_format=data_format)
                elif layer_id.startswith('pool'):
                    current = tf.layers.max_pooling2d(current, params[0], params[1], data_format=data_format)
                else:
                    current = tf.layers.dropout(current, params[0], training=(mode=='train'), seed=seed)
                # layers[layer_id] = current

                # stop if the layer is last (feature layer)
                if layer_id == feat_layer:
                    break

            self.output = current
            if include_top:
                conv10 = tf.layers.conv2d(
                    current, num_classes, 1, kernel_initializer=tf.random_normal_initializer(0.0, 0.01, seed=seed),
                    activation=tf.nn.relu, data_format=data_format, name='conv10'
                ) # discarded in case of finetuning with less than 1000 classes
                axes = [2, 3] if channels_first else [1, 2]
                self.output = tf.reduce_mean(conv10, axes, keepdims=False, name='pool10') # logits
                self.view = conv10


    def load_pretrained_imagenet(self):
        ''' Load network weights and biases (format caffe-tensorflow) pretrained on ImageNet.'''

        self.load_weights('docrec/models/imagenet.npy', ignore_layers=['conv10'], BGR=True, ignore_missing=False)


    def load_weights(self, weights_path, ignore_layers=[], BGR=False, ignore_missing=False):
        ''' Load network weights and biases (format caffe-tensorflow).
        weights_path: path to the numpy-serialized network weights.
        session: current TensorFlow session.
        first_layer: model first layer will be changed in case of BGR data.
        ignore_layers: layers whose parameters must be ignored.
        BGR: if data is BGR, convert weights from the first layer to RGB.
        ignore_missing: if true, serialized weights for missing layers are ignored.
        '''

        first_layer='conv1'
        data_dict = np.load(weights_path, encoding='latin1', allow_pickle=True).item()
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
        ''' Save network weights and biases.
        weights_path: path to the numpy-serialized network weights.
        ignore_layers: layers whose parameters must be ignored.
        '''

        data_dict = {}
        for var in tf.trainable_variables():
            layer, param_name = var.op.name.split('/')[-2 :] # exclude scope if existing
            if layer in ignore_layers:
                continue
            data = self.sess.run(var)
            try:
                data_dict[layer][param_name] = data
            except KeyError:
                data_dict[layer] = {param_name: data}

        # ckeck directory path
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        np.save(weights_path, np.array(data_dict))