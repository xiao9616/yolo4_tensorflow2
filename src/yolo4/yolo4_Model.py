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


def res_sub(x, itea, num_filters):
    x = conv(x, num_filters * 2, (1, 1))
    for i in range(itea):
        x1 = conv(x, num_filters, (1, 1))
        x1 = conv(x1, num_filters, (3, 3))
        x = Add()([x1, x])
    return x


def res(x, itea, num_filters):
    x = conv(x, num_filters * 2, (3, 3), strides=(2, 2))
    x1 = res_sub(x, itea, num_filters * 2)
    x1 = conv(x1, 64, (1, 1))
    x2 = conv(x, 64, (1, 1))
    x = Concatenate()([x1, x2])
    x = conv(x, num_filters * 2, (1, 1))
    return x


def yolo4(input):
    x = conv(input, 32, (3, 3))
    x = res(x, 1, 64)

    x = conv(x, 64, (1, 1))
    x = res(x, )
    x = conv(x, 128, (1, 1))
    x = conv(x, 256, (3, 3), strides=(2, 2))

    x1 = res_sub(x, 8)
    x1 = conv(x1, 128, (1, 1))
    x2 = conv(x, 128, (1, 1))
    x = Concatenate()([x1, x2])

    return x
