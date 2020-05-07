# =============================================
# -*- coding: utf-8 -*-           
# @Time    : 2020/4/25 下午4:02    
# @Author  : xiao9616           
# @Email   : 749935253@qq.com   
# @File    : Mish.py         
# @Software: PyCharm
# ============================================
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
import tensorflow.keras.backend as K


class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
