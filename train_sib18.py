import os
import json
import time
import argparse
import math
import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from docrec.models.squeezenet import SqueezeNet


NUM_CLASSES = 2
NUM_PROC = max(1, multiprocessing.cpu_count() - 1) # for dataset processing


def get_dataset_info(args, mode='train', max_size=None):
    ''' Returns: filenames and labels. '''

    assert mode in ['train', 'val']

    txt_file = '{}/{}.txt'.format(args.samples_dir, mode)
    lines = open(txt_file).readlines()
    if max_size is not None:
        lines = lines [ : int(max_size * len(lines))]
    filenames = []
    labels = []
    for line in lines:
        filename, label = line.split()
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels


def input_fn(args, mode='train', img_shape=(31, 31), resize=False, num_channels=3):
    ''' Dataset load function.'''

    def _parse_function(filename, label):
        ''' Parse function. '''

        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=num_channels, dct_method='INTEGER_ACCURATE') # works with png too
        image = tf.image.convert_image_dtype(image, tf.float32)
        if resize:
            image = tf.image.resize_image_with_crop_or_pad(image, img_shape[0], img_shape[1])
        # one hot representation for training images
        if mode == 'train':
            return image, tf.one_hot(label, NUM_CLASSES)
        return image, label

    assert mode in ['train', 'val']

    filenames, labels = get_dataset_info(args, mode)

    # TF dataset pipeline
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if mode == 'train':
        dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(_parse_function, num_parallel_calls=NUM_PROC)
    dataset = dataset.batch(args.batch_size).repeat(args.num_epochs)
    return dataset.make_one_shot_iterator().get_next()


def train(args):

    ''' Training stage. '''

    # session setup
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # for reproducibility
    np.random.seed(int(args.seed))
    tf.set_random_seed(int(args.seed)) # <= change this in case of multiple runs

    # setting up the dataset of samples
    filenames, _ = get_dataset_info(args, mode='train')
    num_samples = len(filenames)
    input_size = cv2.imread(filenames[0]).shape
    H, W, C = input_size
    resize = False

    # general variables and ops
    global_step_var = tf.Variable(1, trainable=False, name='global_step')
    inc_global_step_op = global_step_var.assign_add(1)

    # placeholders
    images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, H, W, C)) # channels last
    labels_ph = tf.placeholder(tf.float32, name='labels_ph', shape=(None, NUM_CLASSES)) # one-hot enconding

    # dataset iterator
    next_batch_op = input_fn(args, mode='train', img_shape=(H, W), resize=resize, num_channels=C)

    # architecture definition
    model = SqueezeNet(images_ph, num_classes=NUM_CLASSES, mode='train', channels_first=False, sess=sess)
    model_file_ext = 'npy'
    # logits_op = tf.reshape(model.output, [-1, NUM_CLASSES]) # #batches x #classes (squeeze height dimension)
    logits_op = model.output

    # loss function
    loss_op = tf.losses.softmax_cross_entropy(onehot_labels=labels_ph, logits=logits_op)

    # learning rate definition
    num_steps_per_epoch = math.ceil(num_samples / args.batch_size)
    total_steps = args.num_epochs * num_steps_per_epoch
    decay_steps = math.ceil(args.step_size * total_steps)
    learning_rate_op = tf.train.exponential_decay(args.learning_rate, global_step_var, decay_steps, 0.1, staircase=True)

    # optimizer (adam method) and training operation
    optimizer = tf.train.AdamOptimizer(learning_rate_op)
    train_op = optimizer.minimize(loss_op)

    # init graph
    sess.run(tf.global_variables_initializer())

    # pretraining
    model.load_pretrained_imagenet()

    # setup train data directory
    os.makedirs('traindata/{}/model'.format(args.model_id), exist_ok=True)

    # training loop
    start = time.time()
    loss_sample = []
    loss_avg_per_sample = []
    steps = []
    for epoch in range(1, args.num_epochs + 1):
        for step in range(1, num_steps_per_epoch + 1):
            # global step
            global_step = sess.run(global_step_var)
            # batch data
            images, labels = sess.run(next_batch_op)
            # train
            learning_rate, loss, x = sess.run([learning_rate_op, loss_op, train_op], feed_dict={images_ph: images, labels_ph: labels})
            # show training status
            loss_sample.append(loss)
            if (step % 10 == 0) or (step == num_steps_per_epoch):
                loss_avg_per_sample.append(np.mean(loss_sample))
                steps.append(global_step)
                elapsed = time.time() - start
                remaining = elapsed * (total_steps - global_step) / global_step
                print('[{:.2f}%] step={}/{} epoch={} loss={:.5f} :: {:.2f}/{:.2f} seconds lr={}'.format(
                    100 * global_step / total_steps, global_step, total_steps, epoch,
                    loss_avg_per_sample[-1], elapsed, remaining, learning_rate
                ))
                loss_sample = []
            # increment global step
            sess.run(inc_global_step_op)
        # save epoch model
        model.save_weights('traindata/{}/model/{}.{}'.format(args.model_id, epoch, model_file_ext))
    sess.close()

    # save training loss curve
    plt.plot(steps, loss_avg_per_sample)
    plt.savefig('traindata/{}/loss.png'.format(args.model_id))
    loss_plt_data = dict(loss=loss_avg_per_sample, steps=steps)
    np.save('traindata/{}/loss_plt_data.npy'.format(args.model_id), np.array(loss_plt_data))


def validate(args):
    ''' Validate and select the best model. '''

    t0 = time.time() # start cron

    tf.reset_default_graph()

    # session setup
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # K.set_session(sess)

    # setting up the dataset of samples
    filenames, _ = get_dataset_info(args, mode='val')
    num_samples = len(filenames)
    input_size = cv2.imread(filenames[0]).shape
    H, W, C = input_size
    resize = False

    # placeholders
    images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, H, W, C)) # channels last
    labels_ph = tf.placeholder(tf.float32, name='labels_ph', shape=(None,))         # normal enconding

    # dataset iterator
    next_batch_op = input_fn(args, mode='val', img_shape=(H, W), resize=resize, num_channels=C)

    # architecture definition
    model = SqueezeNet(images_ph, num_classes=NUM_CLASSES, mode='val', channels_first=False, sess=sess)
    model_file_ext = 'npy'
    logits_op = tf.reshape(model.output, [-1, NUM_CLASSES]) # #batches x #classes (squeeze height dimension)

    # predictions
    predictions_op = tf.argmax(logits_op, 1)

    # sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    num_steps_per_epoch = math.ceil(num_samples / args.batch_size)
    best_epoch = 0
    best_accuracy = 0.0
    accuracy_per_epoch = []
    for epoch in range(1, args.num_epochs + 1):
        # load epoch model
        model.load_weights('traindata/{}/model/{}.{}'.format(args.model_id, epoch, model_file_ext))
        total_correct = 0
        for step in range(1, num_steps_per_epoch + 1):
            images, labels = sess.run(next_batch_op)
            batch_size = images.shape[0]
            logits, predictions = sess.run([logits_op, predictions_op], feed_dict={images_ph: images, labels_ph: labels})
            num_correct = np.sum(predictions==labels)
            total_correct += num_correct
            if (step % 10 == 0) or (step == num_steps_per_epoch):
                print('step={} accuracy={:.2f}'.format(step, 100 * num_correct / batch_size))
        # epoch average accuracy
        accuracy = 100.0 * total_correct / num_samples
        accuracy_per_epoch.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
        print('-------------------------------------------------')
        print('epoch={} (best={}) accuracy={:.2f} (best={:.2f})'.format(epoch, best_epoch, accuracy, best_accuracy))
        print('-------------------------------------------------')
    sess.close()
    t1 = time.time() # stop cron

    print('best epoch={} accuracy={:.2f}'.format(best_epoch, best_accuracy))
    plt.clf()
    epochs = list(range(1, args.num_epochs + 1))
    plt.plot(epochs, accuracy_per_epoch)
    plt.savefig('traindata/{}/accuracy.png'.format(args.model_id))
    accuracy_plt_data = dict(accuracy=accuracy_per_epoch, epochs=epochs)
    np.save('traindata/{}/accuracy_plt_data.npy'.format(args.model_id), np.array(accuracy_plt_data))
    return best_epoch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training the SqueezeNet (sib18).')
    parser.add_argument(
        '-lr', '--learning-rate', action='store', dest='learning_rate', required=False, type=float,
        default=0.0001, help='Learning rate.'
    )
    parser.add_argument(
        '-bs', '--batch-size', action='store', dest='batch_size', required=False, type=int,
        default=128, help='Batch size.'
    )
    parser.add_argument(
        '-e', '--epochs', action='store', dest='num_epochs', required=False, type=int,
        default=5, help='Number of training epochs.'
    )
    parser.add_argument(
        '-se', '--seed', action='store', dest='seed', required=False, type=float,
        default=100, help='Seed (float) for the training process.'
    )
    parser.add_argument(
        '-ss', '--step-size', action='store', dest='step_size', required=False, type=float,
        default=0.33, help='Step size for learning with step-down policy.'
    )
    parser.add_argument(
        '-sd', '--samples-dir', action='store', dest='samples_dir', required=False, type=str,
        default='~/samples/deeprec-sib18', help='Path where the generated samples were placed.'
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
        'best_model': 'traindata/{}/model/{}.npy'.format(args.model_id, best_epoch),
        'params': args.__dict__
    }
    json.dump(info, open('traindata/{}/info.json'.format(args.model_id), 'w'))
    print('train time={:.2f} min. val time={:.2f} min.'.format(train_time / 60., val_time / 60.))