#!/usr/bin/env python2

'''
Forked from https://github.com/wzgwzg/Multitask_Weather and cleaned to be used
for inference.

This is only for the clear/cloudy sky version. For more details check out
the paper:
    https://www.sciencedirect.com/science/article/pii/S0031320319302481.
'''

import numpy as np
import scipy.io as sio
import tensorflow as tf
import tf_utils as utils
import glob
import json
import os

flags = tf.flags
flags.DEFINE_string('input_dir', 'data', 'Directory containing input images')
flags.DEFINE_integer('batch_size', '4', 'Batch size for input loading')

flags.DEFINE_string('log_dir', 'logs', 'Path to log directory')
flags.DEFINE_string('model_dir', 'model', 'Path to model data')

FLAGS = flags.FLAGS

MODEL_DATA_GDRIVE_URL = (
    'https://drive.google.com/uc?id=1q7OjUZgz2ZzPzfsHPzbNluJye450oid9')
CHECKPOINT_PATH = 'model/MTV4_model.ckpt'
MEAN_MAT_PATH = 'model/mean300.mat'

NUM_OF_CLASSESS = 6
WEATHER_CLASSES = 2
IMAGE_SIZE = 300


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = utils.get_variable(
                np.transpose(kernels, (1, 0, 2, 3)), name=name + '_w')
            bias = utils.get_variable(bias.reshape(-1), name=name + '_b')
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    print('Setting up network...')
    model_data = utils.get_model_data_gdrive(
        FLAGS.model_dir, MODEL_DATA_GDRIVE_URL)

    weights = np.squeeze(model_data['layers'])

    with tf.variable_scope('inference'):
        ''' VGG Base + Convolutional Part of Segmentation Branch '''
        image_net = vgg_net(weights, image)
        conv_final_layer = image_net['conv5_3']

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6_1 = utils.weight_variable([7, 7, 512, 4096], name='W6_1')
        b6_1 = utils.bias_variable([4096], name='b6_1')
        conv6_1 = utils.conv2d_basic(pool5, W6_1, b6_1)
        relu6_1 = tf.nn.relu(conv6_1, name='relu6_1')
        relu_dropout6_1 = tf.nn.dropout(relu6_1, keep_prob=keep_prob)

        W7_1 = utils.weight_variable([1, 1, 4096, 4096], name='W7_1')
        b7_1 = utils.bias_variable([4096], name='b7_1')
        conv7_1 = utils.conv2d_basic(relu_dropout6_1, W7_1, b7_1)
        relu7_1 = tf.nn.relu(conv7_1, name='relu7_1')
        relu_dropout7_1 = tf.nn.dropout(relu7_1, keep_prob=keep_prob)

        W8_1 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS],
                                     name='W8_1')
        b8_1 = utils.bias_variable([NUM_OF_CLASSESS], name='b8_1')
        conv8_1 = utils.conv2d_basic(relu_dropout7_1, W8_1, b8_1)

        ''' Segmentation Branch '''
        deconv_shape1 = image_net['pool4'].get_shape()
        W_t1 = utils.weight_variable(
            [4, 4, deconv_shape1[3], NUM_OF_CLASSESS], name='W_t1')
        b_t1 = utils.bias_variable([deconv_shape1[3]], name='b_t1')
        conv_t1 = utils.conv2d_transpose_strided(
            conv8_1, W_t1, b_t1, output_shape=tf.shape(image_net['pool4']))
        fuse_1 = tf.add(conv_t1, image_net['pool4'], name='fuse_1')

        deconv_shape2 = image_net['pool3'].get_shape()
        W_t2 = utils.weight_variable(
            [4, 4, deconv_shape2[3], deconv_shape1[3]], name='W_t2')
        b_t2 = utils.bias_variable([deconv_shape2[3]], name='b_t2')
        conv_t2 = utils.conv2d_transpose_strided(
            fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net['pool3']))
        fuse_2 = tf.add(conv_t2, image_net['pool3'], name='fuse_2')

        shape = tf.shape(image)
        deconv_shape3 = tf.stack(
            [shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable(
            [16, 16, NUM_OF_CLASSESS, deconv_shape2[3]], name='W_t3')
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name='b_t3')
        conv_t3 = utils.conv2d_transpose_strided(
            fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, axis=3, name='prediction')

        ''' Classification Branch '''
        W6_2 = utils.weight_variable([7, 7, 512, 1024], name='W6_2')
        b6_2 = utils.bias_variable([1024], name='b6_2')
        conv6_2 = utils.conv2d_basic(pool5, W6_2, b6_2)
        relu6_2 = tf.nn.relu(conv6_2, name='relu6_2')
        relu_dropout6_2 = tf.nn.dropout(relu6_2, keep_prob=keep_prob)

        W7_2 = utils.weight_variable([1, 1, 1024, 3840], name='W7_2')
        b7_2 = utils.bias_variable([3840], name='b7_2')
        conv7_2 = utils.conv2d_basic(relu_dropout6_2, W7_2, b7_2)
        relu7_2 = tf.nn.relu(conv7_2, name='relu7_2')
        relu_dropout7_2 = tf.nn.dropout(relu7_2, keep_prob=keep_prob)

        kernel_height = conv7_2.get_shape()[1]
        kernel_width = conv7_2.get_shape()[2]
        conv7_2_gapool = tf.nn.avg_pool(
            relu_dropout7_2,
            ksize=[1, kernel_height, kernel_width, 1],
            strides=[1, kernel_height, kernel_width, 1],
            padding='SAME')

        kernel_height2 = fuse_2.get_shape()[1]
        kernel_width2 = fuse_2.get_shape()[2]
        fuse_2_gapool = tf.nn.avg_pool(
            fuse_2,
            ksize=[1, kernel_height2, kernel_width2, 1],
            strides=[1, kernel_height2, kernel_width2, 1],
            padding='SAME')

        concat_res = tf.concat([conv7_2_gapool, fuse_2_gapool], axis=3)
        concat_res = tf.squeeze(concat_res)

        W8_2 = utils.weight_variable([4096, WEATHER_CLASSES], name='W8_2')
        b8_2 = utils.bias_variable([WEATHER_CLASSES], name='b8_2')
        logits = tf.nn.bias_add(tf.matmul(concat_res, W8_2), b8_2)

    return (tf.expand_dims(annotation_pred, dim=3),
            conv_t3,
            tf.nn.softmax(logits))


def read_and_decode(filename_queue, mean_value):
    image_path = filename_queue.dequeue()

    image = tf.image.decode_jpeg(tf.read_file(image_path))
    image = tf.image.resize_images(image, [300, 300])
    image = tf.reshape(image, [300, 300, 3])
    image -= mean_value
    image, image_path = tf.train.batch([image, image_path],
                                       batch_size=FLAGS.batch_size,
                                       capacity=4000,
                                       num_threads=2)

    return image, image_path


def main(argv=None):
    image_mean_file = sio.loadmat(MEAN_MAT_PATH)
    image_mean_value = image_mean_file['mean']

    keep_probability = tf.placeholder(tf.float32, name='keep_probabilty')
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3],
                           name='input_image')

    segmentation, logits, prob = inference(image, keep_probability)

    print('Setting up saver...')
    saver = tf.train.Saver()

    print('Setting up loaders...')
    input_files = glob.glob(FLAGS.input_dir + '/*.jpg')
    input_filename_queue = tf.train.string_input_producer(input_files)

    input_images, input_image_paths = read_and_decode(
        input_filename_queue, image_mean_value)

    with tf.Session(graph=tf.get_default_graph()) as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver.restore(sess, CHECKPOINT_PATH)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        results = {}
        for k in range(len(input_files) / FLAGS.batch_size):
            input, input_path = sess.run(
                [input_images, input_image_paths]
            )

            pred_segmentation, pred_prob = sess.run(
                [segmentation, prob],
                feed_dict={
                    image: input,
                    keep_probability: 1.0
                })
            input_path = [path.decode('utf-8') for path in input_path]

            for index in range(FLAGS.batch_size):
                mat_path = input_path[index] + '_seg.mat'
                sio.savemat(mat_path, {
                                'mask': pred_segmentation[index].astype(
                                    np.uint8)
                            })

                results[os.path.basename(input_path[index])] = {
                    'segmentation_path': os.path.basename(mat_path),
                    'prob': pred_prob[index].tolist()
                }

                print('Processed ' + input_path[index])

        with open(FLAGS.input_dir + '/predictions.json', 'w') as f:
            json.dump(results, f, indent=2)

        sess.close()
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
