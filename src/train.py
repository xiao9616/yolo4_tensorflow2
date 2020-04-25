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

model_input = Input(shape=(608, 608, 3))
model_output = yolo4(model_input)
model = Model(model_input, model_output)
model.summary()