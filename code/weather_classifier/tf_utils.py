#!/usr/bin/env python2

'''
Forked from https://github.com/wzgwzg/Multitask_Weather and cleaned to be used
for inference.

For more details check out the paper:
    https://www.sciencedirect.com/science/article/pii/S0031320319302481.
'''

import numpy as np
import tensorflow as tf
import scipy.io as sio
import os
import tarfile
import gdown


REQUIRED_MODEL_FILES = [
    'MTV4_model.ckpt.data-00000-of-00001',
    'MTV4_model.ckpt.meta',
    'imagenet-vgg-verydeep-19.mat',
    'MTV4_model.ckpt.index',
    'mean300.mat'
]
VGG_MODEL_MAT = 'imagenet-vgg-verydeep-19.mat'
MODEL_DATA_ARCHIVE = 'MULTI_TASK_WEATHER_MODEL.tar.gz'


def get_model_data_gdrive(dir_path, model_gdrive_url):
    maybe_download_and_extract(dir_path, model_gdrive_url)

    vgg_mat_path = os.path.join(dir_path, VGG_MODEL_MAT)

    if not os.path.exists(vgg_mat_path):
        raise IOError("VGG Model not found!")

    return sio.loadmat(vgg_mat_path)


def maybe_download_and_extract(dir_path, gdrive_url):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    download = not all(
        os.path.exists(os.path.join(dir_path, file))
        for file in REQUIRED_MODEL_FILES
    )

    if download:
        gdown.download(gdrive_url, MODEL_DATA_ARCHIVE, quiet=False)
        tarfile.open(MODEL_DATA_ARCHIVE, 'r:gz').extractall(dir_path)


def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    if name[-1] == 'w':
        var = tf.get_variable(
            name=name,
            initializer=init,
            regularizer=tf.contrib.layers.l2_regularizer(0.001),
            shape=weights.shape)
    else:
        var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var


def weight_variable(shape, stddev=0.02, name=None):
    if name is None:
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial)
    else:
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(0.001))


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def conv2d_basic(x, W, bias, mypadding='SAME'):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=mypadding)
    return tf.nn.bias_add(conv, bias)


def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv2d_transpose_strided(x, W, b, output_shape=None, stride=2):
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]

    conv = tf.nn.conv2d_transpose(
        x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def leaky_relu(x, alpha=0.0, name=""):
    return tf.maximum(alpha * x, x, name)


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)
