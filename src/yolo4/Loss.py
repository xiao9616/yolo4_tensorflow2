# =============================================
# -*- coding: utf-8 -*-           
# @Time    : 2020/5/19 下午3:18    
# @Author  : xiao9616           
# @Email   : 749935253@qq.com   
# @File    : Loss.py         
# @Software: PyCharm
# @Discript:
# ============================================
from tensorflow.keras import losses
from src.yolo4.config import *
from tensorflow.keras import backend as K
from src.yolo4.util import *


class YOLO4_LOSS(losses.Loss):
    def __init__(self):
        super().__init__()
        self.scale_num = len(scale_size)

    def call(self, y_true, y_pred):
        loss = self.__calculate_loss(y_true, y_pred)
        return loss

    def __calculate_loss(self, y_true, y_pred, ignore_thresh=0.5, label_smoothing=True, use_focal_loss=False,
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
        anchors = get_anchors(anchors_path)
        num_layers = len(anchors) // 3
        yolo_outputs = y_pred
        y_true = y_true
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
                                                          anchors[anchor_mask[l]], num_classes, input_shape,
                                                          calc_loss=True)
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
                confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                      from_logits=True) + \
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
                    class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:],
                                                                     from_logits=True)

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
        # loss = K.expand_dims(loss, axis=-1)
        return loss
