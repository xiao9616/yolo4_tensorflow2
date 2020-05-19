# =============================================
# -*- coding: utf-8 -*-           
# @Time    : 2020/4/25 下午3:44    
# @Author  : xiao9616           
# @Email   : 749935253@qq.com   
# @File    : train.py         
# @Software: PyCharm
# ============================================
from src.yolo4.util import *
from src.yolo4.Net import *
from configparser import ConfigParser
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

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
weights_path = config.get("filepath", "weights_path")


def train(batch_size, annotation_path, classes_path, anchors_path, weights_path, log_path, optimizer=Adam(lr=1e-3),
          epochs=50):
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (608, 608)  # multiple of 32, hw

    model = get_yolo4_model(input_shape, anchors, num_classes, weights_path=weights_path,
                            load_pretrained=False)  # make sure you know what you freeze
    model.fit()
    logging = TensorBoard(log_dir=log_path)
    checkpoint = ModelCheckpoint(log_path + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=optimizer, loss={'yolo4_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors,
                                                           num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=epochs,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])
        model.save_weights(log_path + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=optimizer,
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 4  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors,
                                                           num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=100,
                            initial_epoch=50,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_path + 'trained_weights_final.h5')


if __name__ == '__main__':
    yolo4_model = yolo4(Input(shape=(608, 608, 3)), anchors_num, classes_num)
    yolo4_model.summary()
    train(32, test_path, classname_path, anchors_path, weights_path, log_path)
