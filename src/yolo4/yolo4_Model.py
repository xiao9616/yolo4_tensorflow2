# =============================================
# -*- coding: utf-8 -*-           
# @Time    : 2020/4/25 下午3:44    
# @Author  : xiao9616           
# @Email   : 749935253@qq.com   
# @File    : yolo4_Model.py         
# @Software: PyCharm
# ============================================
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation, MaxPool2D, UpSampling2D
from tensorflow.keras.activations import linear
from .Mish import *


def conv(x, filters, kernel_size, strides=(1, 1), padding='same', activation="Mish", use_bias=True):
    x = Conv2D(filters, kernel_size, strides, padding, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    if activation == 'LeakRelu':
        x = LeakyReLU()(x)
    if activation == 'Mish':
        x = Activation('Mish')(x)
    if activation == 'Linear':
        x = linear(x)
    return x


def res(x, nf_in, itea_sub, nf_sub, nf_left, nf_right, nf_out):
    x = conv(x, nf_in, (3, 3), strides=(2, 2))
    x_sub = res_sub(x, itea_sub, nf_sub)
    x_left = conv(x_sub, nf_left, (1, 1))
    x_right = conv(x, nf_right, (1, 1))
    x = Concatenate()([x_left, x_right])
    x = conv(x, nf_out, (1, 1))
    return x


def res_sub(x, itea, num_filters):
    x = conv(x, num_filters, (1, 1))
    for i in range(itea):
        x1 = conv(x, num_filters, (1, 1))
        x1 = conv(x1, num_filters, (3, 3))
        x = Add()([x1, x])
    return x


def spp_sub(x):
    x1 = MaxPool2D(strides=(1, 1), pool_size=(3, 3), padding='same')(x)
    x2 = MaxPool2D(strides=(1, 1), pool_size=(9, 9), padding='same')(x)
    x3 = MaxPool2D(strides=(1, 1), pool_size=(13, 13), padding='same')(x)
    x_out = Concatenate()([x1, x2, x3, x])
    return x_out


def upper_concate(x1, x2, num_filter1, num_filter2):
    x1 = conv(x1, num_filter1, (1, 1), activation='LeakRelu')
    x2 = conv(x2, num_filter1, (1, 1), activation='LeakRelu')
    x2 = UpSampling2D()(x2)
    x = Concatenate()([x1, x2])
    x = conv(x, num_filter1, (1, 1), activation='LeakRelu')
    x = conv(x, num_filter2, (3, 3), activation='LeakRelu')
    x = conv(x, num_filter1, (1, 1), activation='LeakRelu')
    x = conv(x, num_filter2, (3, 3), activation='LeakRelu')
    x = conv(x, num_filter1, (1, 1), activation='LeakRelu')
    return x


def spp(x):
    x = conv(x, 512, (1, 1), activation='LeakRelu')
    x = conv(x, 1024, (3, 3), activation='LeakRelu')
    x = conv(x, 512, (1, 1), activation='LeakRelu')
    x = spp_sub(x)
    x = conv(x, 512, (1, 1), activation='LeakRelu')
    x = conv(x, 1024, (1, 1), activation='LeakRelu')
    x = conv(x, 512, (1, 1), activation='LeakRelu')
    return x


def yolo(x, num_filter1, num_filter2):
    x = conv(x, num_filter1, (3, 3), activation='LeakRelu')
    x = conv(x, num_filter2, (1, 1), activation='Linear')
    return x


def merge(x1, x2, num_filter1, num_filter2):
    x1 = conv(x1, num_filter1, (3, 3), strides=(2, 2), activation='LeakRule')
    x = Concatenate()([x1, x2])
    x = conv(x, num_filter1, (1, 1), activation='LeakRelu')
    x = conv(x, num_filter2, (3, 3), activation='LeakRelu')
    x = conv(x, num_filter1, (1, 1), activation='LeakRelu')
    x = conv(x, num_filter2, (3, 3), activation='LeakRelu')
    x = conv(x, num_filter1, (1, 1), activation='LeakRelu')

    return x


def yolo4(input, anchor_nums, classes_num):
    out_shape = anchor_nums * (5 + classes_num)
    x = conv(input, 32, (3, 3))
    x = res(x, 64, 1, 64, 64, 64, 64)
    x = res(x, 128, 2, 64, 64, 64, 128)
    x1 = res(x, 128, 8, 128, 128, 128, 256)
    x2 = res(x1, 256, 8, 256, 256, 256, 512)
    x3 = res(x2, 1024, 4, 512, 512, 512, 1024)
    x3 = spp(x3)
    x2 = upper_concate(x2, x3, 256, 512)
    x1 = upper_concate(x1, x2, 128, 256)
    yolo1 = yolo(x1, 256, out_shape)
    x2 = merge(x1, x2, 256, 512)
    yolo2 = yolo(x2, 512, out_shape)
    x3 = merge(x2, x3, 512, 1024)
    yolo3 = yolo(x3, 1024, out_shape)
    return yolo1, yolo2, yolo3
