# =============================================
# -*- coding: utf-8 -*-           
# @Time    : 2020/4/25 下午3:44    
# @Author  : xiao9616           
# @Email   : 749935253@qq.com   
# @File    : train.py         
# @Software: PyCharm
# ============================================

from src.yolo4.util import *
from configparser import ConfigParser
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

config = ConfigParser()
config.read("./config.ini")

input_shape = int(config.get("model", "input_shape"))
anchors_num = int(config.get("model", "anchors_num"))
classes_num = int(config.get("data", "classes_num"))

anchors_path = config.get("filepath", "anchors_path")
train_path = config.get("filepath", "train_path")
eval_path = config.get("filepath", "eval_path")
test_path = config.get("filepath", "test_path")
train_eval_path = config.get("filepath", "train_eval_path")
classname_path = config.get("filepath", "classname_path")
log_path = config.get("filepath", "log_path")


def train(batch_size=32, optimizer=Adam(1e-3), epochs=1000):
    log_dir = log_path + "yolo4_log/"
    logging = TensorBoard(log_dir=log_dir)
    model = get_yolo4_model((input_shape, input_shape, 3), anchors_path, classes_num)

    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    anchors = get_anchors(anchors_path)
    with open(train_path) as f:
        train_lines = f.readlines()

    with open(eval_path) as f:
        eval_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(eval_lines)
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        model.fit_generator(data_generator(train_lines[:], batch_size, input_shape, anchors, classes_num),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator(
                                eval_lines[:], batch_size, input_shape, anchors, classes_num),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=epochs,
                            initial_epoch=50,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')


if __name__ == '__main__':
    train()
