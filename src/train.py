# =============================================
# -*- coding: utf-8 -*-           
# @Time    : 2020/4/25 下午3:44    
# @Author  : xiao9616           
# @Email   : 749935253@qq.com   
# @File    : train.py         
# @Software: PyCharm
# ============================================

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from src.yolo4.yolo4_Model import yolo4
from configparser import ConfigParser

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

model_input = Input(shape=(input_shape, input_shape, 3))
model_output = yolo4(model_input, anchors_num, classes_num)
model = Model(model_input, model_output)
model.summary()
