# =============================================
# -*- coding: utf-8 -*-           
# @Time    : 2020/4/25 下午3:44    
# @Author  : xiao9616           
# @Email   : 749935253@qq.com   
# @File    : yolo4_Model.py         
# @Software: PyCharm
# ============================================
import tensorflow as tf
import numpy as np

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

NUM_CLASS = 80
ANCHORS = np.array([1.25, 1.625, 2.0, 3.75, 4.125, 2.875, 1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375, 3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875], dtype=np.float32).reshape(3, 3, 2)
STRIDES = np.array([8, 16, 32])
IOU_LOSS_THRESH = 0.5


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


def decode(conv_output, i=0):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """

    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def bbox_iou(boxes1, boxes2):

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_loss(pred, conv, label, bboxes, i=0):

    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss

