# -*- coding: utf-8 -*-

import tensorflow as tf
from .Mish import *

def convolutional(input_layer,
                  filters_shape,
                  func,
                  down_sample=False,
                  activate=True,
                  batch_normalization=True):
    """
    卷积层：yolo单层卷积设置；对偏置值进行调整，在常规卷积阶段不使用；激活函数中backbone使用Mish，后续层使用leaky
    """
    if down_sample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], 
                                  kernel_size = filters_shape[0],
                                  strides=strides,
                                  padding=padding,
                                  use_bias=not batch_normalization,
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if batch_normalization:
        conv = tf.keras.layers.BatchNormalization()(conv)
    if activate:
        conv = Activation('Mish')(conv) if func == 'Mish' else tf.nn.leaky_relu(conv, alpha=0.1)
    
    return conv

def residual_block(input_layer, input_channal, filter_1, filter_2):
    """
    yolo网络中残差块，由两个卷积层构成，与输入层进行混叠
    """
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channal, filter_1))
    conv = convolutional(conv, filters_shape=(3, 3, filter_1, filter_2))

    residual_output = short_cut + conv
    return residual_output

def backbone():
    """
    darknet53网络结构，一共五层，分为两路：一路包含n个残差块，另一路直接卷积，最后两路结果进行拼接。残差块数量n分别为1、2、8、8、4。
    """