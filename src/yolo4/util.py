# =============================================
# -*- coding: utf-8 -*-           
# @Time    : 2020/4/28 下午2:33    
# @Author  : xiao9616           
# @Email   : 749935253@qq.com   
# @File    : util.py         
# @Software: PyCharm
# ============================================
import math
import matplotlib.colors
from .Net import *
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
import tensorflow.keras.backend as K
from functools import reduce
from PIL import Image
import numpy as np


class IOUSameXY():
    def __init__(self, anchors, boxes):
        super(IOUSameXY, self).__init__()
        self.anchor_max = anchors / 2
        self.anchor_min = - self.anchor_max
        self.box_max = boxes / 2
        self.box_min = - self.box_max
        self.anchor_area = anchors[..., 0] * anchors[..., 1]
        self.box_area = boxes[..., 0] * boxes[..., 1]

    def calculate_iou(self):
        intersect_min = np.maximum(self.box_min, self.anchor_min)
        intersect_max = np.minimum(self.box_max, self.anchor_max)
        intersect_wh = np.maximum(intersect_max - intersect_min + 1.0, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # w * h
        union_area = self.anchor_area + self.box_area - intersect_area
        iou = intersect_area / union_area  # shape : [N, 9]

        return iou


class ReadTxt(object):
    def __init__(self, line_bytes):
        super(ReadTxt, self).__init__()
        # bytes -> string
        self.line_str = bytes.decode(line_bytes, encoding="utf-8")

    def parse_line(self):
        line_info = self.line_str.strip('\n')
        split_line = line_info.split(" ")
        box_num = (len(split_line) - 1) / 5
        image_name = split_line[0]
        # print("Reading {}".format(image_name))
        split_line = split_line[1:]
        boxes = []
        for i in range(MAX_TRUE_BOX_NUM_PER_IMG):
            if i < box_num:
                box_xmin = int(float(split_line[i * 5]))
                box_ymin = int(float(split_line[i * 5 + 1]))
                box_xmax = int(float(split_line[i * 5 + 2]))
                box_ymax = int(float(split_line[i * 5 + 3]))
                class_id = int(split_line[i * 5 + 4])
                boxes.append([box_xmin, box_ymin, box_xmax, box_ymax, class_id])
            else:
                box_xmin = 0
                box_ymin = 0
                box_xmax = 0
                box_ymax = 0
                class_id = 0
                boxes.append([box_xmin, box_ymin, box_xmax, box_ymax, class_id])

        return image_name, boxes


class GenerateLabel():
    def __init__(self, true_boxes, input_shape):
        super(GenerateLabel, self).__init__()
        self.true_boxes = np.array(true_boxes, dtype=np.float32)
        self.input_shape = np.array(input_shape, dtype=np.int32)
        self.anchors = np.array(anchor, dtype=np.float32)
        self.batch_size = self.true_boxes.shape[0]

    def generate_label(self):
        center_xy = (self.true_boxes[..., 0:2] + self.true_boxes[..., 2:4]) // 2  # shape : [B, N, 2]
        box_wh = self.true_boxes[..., 2:4] - self.true_boxes[..., 0:2]  # shape : [B, N, 2]
        self.true_boxes[..., 0:2] = center_xy / self.input_shape  # Normalization
        self.true_boxes[..., 2:4] = box_wh / self.input_shape  # Normalization
        true_label_1 = np.zeros(
            (self.batch_size, scale_size[0], scale_size[0], 3, classes_num + 5))
        true_label_2 = np.zeros(
            (self.batch_size, scale_size[1], scale_size[1], 3, classes_num + 5))
        true_label_3 = np.zeros(
            (self.batch_size, scale_size[2], scale_size[2], 3, classes_num + 5))
        # true_label : list of 3 arrays of type numpy.ndarray(all elements are 0), which shapes are:
        # (self.batch_size, 13, 13, 3, 5 + C)
        # (self.batch_size, 26, 26, 3, 5 + C)
        # (self.batch_size, 52, 52, 3, 5 + C)
        true_label = [true_label_1, true_label_2, true_label_3]
        # shape : (9, 2) --> (1, 9, 2)
        anchors = np.expand_dims(self.anchors, axis=0)
        # valid_mask filters out the valid boxes.
        valid_mask = box_wh[..., 0] > 0

        for b in range(self.batch_size):
            wh = box_wh[b, valid_mask[b]]
            if len(wh) == 0:
                # For pictures without boxes, iou is not calculated.
                continue
            # shape of wh : [N, 1, 2], N is the actual number of boxes per picture
            wh = np.expand_dims(wh, axis=1)
            # Calculate the iou between the box and the anchor, both center points are (0, 0).
            iou_value = IOUSameXY(anchors=anchors, boxes=wh).calculate_iou()
            # shape of best_anchor : [N]
            best_anchor = np.argmax(iou_value, axis=-1)
            for i, n in enumerate(best_anchor):
                for s in range(3):
                    if n in anchor_index[s]:
                        x = np.floor(self.true_boxes[b, i, 0] * scale_size[s]).astype('int32')
                        y = np.floor(self.true_boxes[b, i, 1] * scale_size[s]).astype('int32')
                        anchor_id = anchor_index[s].index(n)
                        class_id = self.true_boxes[b, i, 4].astype('int32')
                        true_label[s][b, y, x, anchor_id, 0:4] = self.true_boxes[b, i, 0:4]
                        true_label[s][b, y, x, anchor_id, 4] = 1
                        true_label[s][b, y, x, anchor_id, 5 + class_id - 1] = 1

        return true_label


def parse_dataset_batch(dataset):
    image_name_list = []
    boxes_list = []
    len_of_batch = dataset.shape[0]
    for i in range(len_of_batch):
        image_name, boxes = ReadTxt(line_bytes=dataset[i].numpy()).parse_line()
        image_name_list.append(image_name)
        boxes_list.append(boxes)
    boxes_array = np.array(boxes_list)
    return image_name_list, boxes_array


def generate_label_batch(true_boxes):
    true_label = GenerateLabel(true_boxes=true_boxes, input_shape=[input_shape[0], input_shape[1]]).generate_label()
    return true_label


def process_image_batch(filenames):
    image_list = []
    for filename in filenames:
        image_tensor = process_single_image(filename)
        image_list.append(image_tensor)
    return tf.concat(values=image_list, axis=0)


def process_single_image(image_filename):
    img_raw = tf.io.read_file(image_filename)
    image = tf.io.decode_jpeg(img_raw, channels=3)
    image = resize_image_with_pad(image=image)
    image = tf.dtypes.cast(image, dtype=tf.dtypes.float32)
    image = image / 255.0
    return image


def resize_image_with_pad(image):
    image_tensor = tf.image.resize_with_pad(image=image, target_height=input_shape[0],
                                            target_width=input_shape[0])
    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor = image_tensor / 255.0
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor


def get_yolo4_model(input_shape, anchors, num_classes, weights_path, load_pretrained=True, freeze_body=2):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo4(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (250, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    label_smoothing = 0
    use_focal_obj_loss = False
    use_focal_loss = False
    use_diou_loss = True
    use_softmax_loss = False

    model_loss = Lambda(yolo4_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5,
                                   'label_smoothing': label_smoothing, 'use_focal_obj_loss': use_focal_obj_loss,
                                   'use_focal_loss': use_focal_loss, 'use_diou_loss': use_diou_loss,
                                   'use_softmax_loss': use_softmax_loss})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = tf.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = tf.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_giou(b1, b2):
    """
    Calculate GIoU loss on anchor boxes
    Reference Paper:
        "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
        https://arxiv.org/abs/1902.09630

    Parameters
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    Returns
    -------
    giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + tf.keras.backend.epsilon())

    # get enclosed area
    enclose_mins = tf.minimum(b1_mins, b2_mins)
    enclose_maxes = tf.maximum(b1_maxes, b2_maxes)
    enclose_wh = tf.maximum(enclose_maxes - enclose_mins, 0.0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    # calculate GIoU, add epsilon in denominator to avoid dividing by 0
    giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + tf.keras.backend.epsilon())
    giou = tf.expand_dims(giou, -1)

    return giou


def box_diou(b1, b2):
    """
    Calculate DIoU loss on anchor boxes
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    Returns
    -------
    diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + tf.keras.backend.epsilon())

    # box center distance
    center_distance = tf.keras.backend.sum(tf.square(b1_xy - b2_xy), axis=-1)
    # get enclosed area
    enclose_mins = tf.minimum(b1_mins, b2_mins)
    enclose_maxes = tf.maximum(b1_maxes, b2_maxes)
    enclose_wh = tf.maximum(enclose_maxes - enclose_mins, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = tf.keras.backend.sum(tf.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + tf.keras.backend.epsilon())

    # calculate param v and alpha to extend to CIoU
    # v = 4*K.square(tf.math.atan2(b1_wh[..., 0], b1_wh[..., 1]) - tf.math.atan2(b2_wh[..., 0], b2_wh[..., 1])) / (math.pi * math.pi)
    # alpha = v / (1.0 - iou + v)
    # diou = diou - alpha*v

    diou = tf.expand_dims(diou, -1)
    return diou


def box_ciou(b1, b2):
    """
    Parameters
    ----------
    b1: tensor, shape = (batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape = (batch, feat_w, feat_h, anchor_num, 4), xywh

    Returns
    -------
    ciou: tensor, shape = (batch, feat_w, feat_h, anchor_num, 1)
    """
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + tf.keras.backend.epsilon())

    # box center distance
    center_distance = tf.keras.backend.sum(tf.square(b1_xy - b2_xy), axis=-1)
    # get enclosed area
    enclose_mins = tf.minimum(b1_mins, b2_mins)
    enclose_maxes = tf.maximum(b1_maxes, b2_maxes)
    enclose_wh = tf.maximum(enclose_maxes - enclose_mins, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = tf.keras.backend.sum(tf.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + tf.keras.backend.epsilon())

    # calculate param v and alpha to extend to CIoU
    v = 4 * tf.keras.backend.square(
        tf.math.atan2(b1_wh[..., 0], b1_wh[..., 1]) - tf.math.atan2(b2_wh[..., 0], b2_wh[..., 1])) / (math.pi * math.pi)
    alpha = v / (1.0 - iou + v)
    ciou = diou - alpha * v

    ciou = tf.expand_dims(ciou, -1)
    return ciou


def smooth_labels(y_true, label_smoothing):
    label_smoothing = tf.constant(label_smoothing, dtype=tf.float16)
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing


def yolo4_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """
    reshape yolo4的输出特征并将box，confidence，classId归一化
    Args:
        feats: 
        anchors: 
        num_classes: 
        input_shape: 
        calc_loss: 

    Returns:

    """

    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = tf.shape(feats)[1:3]  # height, width
    grid_y = tf.tile(tf.reshape(tf.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                     [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                     [grid_shape[0], 1, 1, 1])
    grid = tf.concatenate([grid_x, grid_y])
    grid = tf.cast(grid, tf.dtype(feats))

    feats = tf.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[::-1], tf.dtype(feats))
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.dtype(feats))
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """
    Preprocess true boxes to training input format
    Args:
        true_boxes: array, shape=(m, T, 5) Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
        input_shape: array-like, hw, multiples of 32
        anchors: array, shape=(N, 2), wh
        num_classes: integer

    Returns:
        y_true: list of array, shape like yolo_outputs, xywh are reletive value

    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  # 中心点坐标
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # w，h
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]  # 相对坐标
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def softmax_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Compute softmax focal loss.
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.

    # Returns
        softmax_focal_loss: Softmax focal loss, tensor of shape (?, num_boxes).
    """

    # Scale predictions so that the class probas of each sample sum to 1
    # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    # Clip the prediction value to prevent NaN's and Inf's
    # epsilon = K.epsilon()
    # y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)

    # Calculate Cross Entropy
    cross_entropy = -y_true * tf.math.log(y_pred)

    # Calculate Focal Loss
    softmax_focal_loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy

    return softmax_focal_loss


def sigmoid_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Compute sigmoid focal loss.
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.

    # Returns
        sigmoid_focal_loss: Sigmoid focal loss, tensor of shape (?, num_boxes).
    """
    sigmoid_loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)

    pred_prob = tf.sigmoid(y_pred)
    p_t = ((y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob)))
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

    sigmoid_focal_loss = modulating_factor * alpha_weight_factor * sigmoid_loss
    # sigmoid_focal_loss = tf.reduce_sum(sigmoid_focal_loss, axis=-1)

    return sigmoid_focal_loss


def yolo4_loss(args, anchors, num_classes, ignore_thresh=0.5, label_smoothing=True, use_focal_loss=False,
               use_focal_obj_loss=False, use_softmax_loss=False, use_giou_loss=False, use_diou_loss=False,
               use_ciou_loss=True):
    """
    定义yolo4损失
    Args:
        args:args是Lambda层的输入，即model_body.output和y_true的组合
        anchors:二维数组，结构是(9, 2)，即9个anchor box
        num_classes:类别数
        ignore_thresh:置信度过滤阈值
        label_smoothing:标签平滑系数

        use_focal_loss:是否使用focal损失
        use_focal_obj_loss:
        use_softmax_loss:
        use_giou_loss:
        use_ciou_loss:是否使用ciou损失

    Returns:
        loss：所有的损失和
    """

    # num_layers：层的数量，是anchors数量的3分之1；
    # yolo_outputs和y_true：分离args，前3个是yolo_outputs预测值，后3个是y_true真值；
    # anchor_mask：anchor box的索引数组，3个1组倒序排序，即[[6, 7, 8], [3, 4, 5], [0, 1, 2]]；
    # input_shape：K.shape(yolo_outputs[0])[1:3]，第1个预测矩阵yolo_outputs[0]的结构（shape）的第1~2位，即(?, 13, 13, 18)中的(13, 13)。再x32，就是YOLO网络的输入尺寸，即(416, 416)，因为在网络中，含有5个步长为(2, 2)的卷积操作，降维32 = 5 ^ 2倍；
    # grid_shapes：与input_shape类似，K.shape(yolo_outputs[l])[1:3]，以列表的形式，选择3个尺寸的预测图维度，即[(13, 13), (26, 26), (52, 52)]；
    # m：第1个预测图的结构的第1位，即K.shape(yolo_outputs[0])[0]，输入模型的图片总量，即批次数；
    # mf：m的float类型，即K.cast(m, K.dtype(yolo_outputs[0]))
    # loss：损失值为0；

    num_layers = len(anchors) // 3
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[i])[1:3] * 32, K.dtype(y_true[0])) for i in range(num_layers)]
    m = K.shape(yolo_outputs[0])[0]
    mf = K.cast(m, K.dtype(yolo_outputs[0]))
    loss = 0
    total_location_loss = 0
    total_confidence_loss = 0
    total_class_loss = 0
    for l in rand(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = smooth_labels(true_class_probs, label_smoothing)

        grid, raw_pred, pred_xy, pred_wh = yolo4_head(yolo_outputs[l],
                                                      anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        if use_focal_obj_loss:
            # Focal loss for objectness confidence
            confidence_loss = sigmoid_focal_loss(object_mask, raw_pred[..., 4:5])
        else:
            confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                              (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                        from_logits=True) * ignore_mask

        if use_focal_loss:
            # Focal loss for classification score
            if use_softmax_loss:
                class_loss = softmax_focal_loss(true_class_probs, raw_pred[..., 5:])
            else:
                class_loss = sigmoid_focal_loss(true_class_probs, raw_pred[..., 5:])
        else:
            if use_softmax_loss:
                # use softmax style classification output
                class_loss = object_mask * K.expand_dims(
                    K.categorical_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True), axis=-1)
            else:
                # use sigmoid style classification output
                class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        if use_giou_loss:
            # Calculate GIoU loss as location loss
            raw_true_box = y_true[l][..., 0:4]
            giou = box_giou(pred_box, raw_true_box)
            giou_loss = object_mask * box_loss_scale * (1 - giou)
            giou_loss = K.sum(giou_loss) / mf
            location_loss = giou_loss
        elif use_diou_loss:
            # Calculate DIoU loss as location loss
            raw_true_box = y_true[l][..., 0:4]
            diou = box_diou(pred_box, raw_true_box)
            diou_loss = object_mask * box_loss_scale * (1 - diou)
            diou_loss = K.sum(diou_loss) / mf
            location_loss = diou_loss
        elif use_ciou_loss:
            # Calculate CIoU loss as location loss
            raw_true_box = y_true[l][..., 0:4]
            ciou = box_ciou(pred_box, raw_true_box)
            ciou_loss = object_mask * box_loss_scale * (1 - ciou)
            ciou_loss = K.sum(ciou_loss) / mf
            location_loss = ciou_loss
        else:
            # Standard YOLO location loss
            # K.binary_crossentropy is helpful to avoid exp overflow.
            xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                           from_logits=True)
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
            xy_loss = K.sum(xy_loss) / mf
            wh_loss = K.sum(wh_loss) / mf
            location_loss = xy_loss + wh_loss

        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += location_loss + confidence_loss + class_loss
        total_location_loss += location_loss
        total_confidence_loss += confidence_loss
        total_class_loss += class_loss

    # Fit for tf 2.0.0 loss shape
    loss = K.expand_dims(loss, axis=-1)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5,
                    proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes: box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = matplotlib.colors.rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = matplotlib.hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    while True:
        image_data = []
        box_data = []
        i = 0
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines, input_shape)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=100,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
