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
from tensorflow.keras.layers import Activation
from .Mish import *


def conv(x,
         filters,
         kernel_size,
         strides=(1, 1),
         padding='same',
         data_format=None,
         dilation_rate=(1, 1),
         activation=None,
         use_bias=True,
         kernel_initializer='glorot_uniform',
         bias_initializer='zeros',
         kernel_regularizer=None,
         bias_regularizer=None,
         activity_regularizer=None,
         kernel_constraint=None,
         bias_constraint=None,
         **kwargs):
    x = Conv2D(filters, kernel_size, strides, padding, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    x = Activation('Mish')(x)
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


def yolo4(input):
    x = conv(input, 32, (3, 3))
    x = res(x, 64, 1, 64, 64, 64, 64)
    x = res(x, 128, 2, 64, 64, 64, 128)
    x = res(x, 128, 8, 128, 128, 128, 256)

    return x
