# =============================================
# -*- coding: utf-8 -*-           
# @Time    : 2020/5/14 上午10:50    
# @Author  : xiao9616           
# @Email   : 749935253@qq.com   
# @File    : BaseModel.py
# @Software: PyCharm
# ============================================

import logging
import tensorflow as tf
import os
from src.yolo4.config import *
from src.yolo4.util import *
from src.yolo4.Net import YOLO4_NET
from src.yolo4.Loss import YOLO4_LOSS

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename="./yolo4/logs/train.log", filemode='w+')


class BaseModel(object):
    '''
    一个自定义的类，需要重写方法：

    '''

    def data_generator(self):
        '''

        Returns:该方法可以重写, 并且返回一个tf.data对象

        '''
        txt_data = tf.data.TextLineDataset(filenames=train_path)
        count = 0
        for _ in txt_data:
            count += 1
        train_data = txt_data.batch(batch_size=batch_size)

        return train_data, count

    def net_generator(self):
        net = YOLO4_NET()
        return net

    def loss_generator(self):
        loss = YOLO4_LOSS()
        return loss

    def optimizer_generator(self):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=3000,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        return optimizer

    def metric_generator(self):
        metric = tf.keras.metrics.Mean()
        return metric

    def train(self):
        # GPU 设置
        tf.debugging.set_log_device_placement(True)
        if use_gpu:
            gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
            if gpus:
                logging.info("use gpu device")
                # gpu显存分配
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(device=gpu, enable=True)
                    tf.print(gpu)
            else:
                os.environ["CUDA_VISIBLE_DEVICE"] = "-1"
                logging.info("not found gpu device,convert to use cpu")
        else:
            logging.info("use cpu device")
            # 禁用gpu
            os.environ["CUDA_VISIBLE_DEVICE"] = "-1"

        # 训练数据
        train_dataset, train_count = self.data_generator()
        # 网络结构
        net = self.net_generator()
        net.summary()

        global fine_tune_epoch
        # 是否finetune
        if fine_tune:
            net.load_weights(filepath=weights_path + "epoch-{}".format(fine_tune_epoch))
            print("load {} epoch weigth".format(fine_tune))
        else:

            fine_tune_epoch = -1
            print("train model from init")

        # 设置loss损失函数
        loss = self.loss_generator()

        # 设置优化器optimizer
        optimizer = self.optimizer_generator()

        # 设置评价指标
        metric = self.metric_generator()

        # 模型训练与更新
        for epoch in range(fine_tune_epoch + 1, train_epochs):
            step = 0
            for train_dataset_batch in train_dataset:
                # print(train_dataset_batch)
                step += 1
                images, boxes = parse_dataset_batch(dataset=train_dataset_batch)
                image_batch = process_image_batch(images)
                label_batch = generate_label_batch(boxes)
                with tf.GradientTape() as tape:
                    out = net(image_batch, trainable=True)
                    total_loss = loss(y_true=label_batch, y_pred=out)
                gradients = tape.gradient(total_loss, net.trainable_variables)
                optimizer.apply_gradients(grads_and_vars=zip(gradients, net.trainable_variables))
                metric.updates(values=total_loss)
                print("Epoch: {}/{}, step: {}/{} ,loss: {:.5f}".format(
                    epoch, train_epochs, step, tf.math.ceil(train_count / batch_size), metric.result()
                ))
            metric.reset_states()
            if epoch % save_frequency == 0:
                net.save_weights(filepath=weights_path + "epoch-{}".format(epoch), save_format='tf')
        net.save_weights(filepath=weights_path + "epoch-{}".format(train_epochs), save_format='tf')


if __name__ == '__main__':
    yolo = BaseModel()
    yolo.train()
