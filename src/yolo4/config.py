# =============================================
# -*- coding: utf-8 -*-           
# @Time    : 2020/5/18 上午10:31    
# @Author  : xiao9616           
# @Email   : 749935253@qq.com   
# @File    : config.py
# @Software: PyCharm
# @Discript:
# ============================================
input_shape = (604, 604, 3)
anchors_num = 3
classes_num = 80
anchor = [[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119], [10, 13], [16, 30], [33, 23]]
anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
scale_size = [13, 26, 52]
MAX_TRUE_BOX_NUM_PER_IMG = 20
root_path = "/home/user/github/yolo4_tensorflow2/"

anchors_path = root_path + "sources/dataSet/train_anchor/anchors.txt"
classname_path = root_path + "sources/dataSet/Annotations/classes.txt"

train_path = root_path + "sources/dataSet/train_file/train.txt"
eval_path = root_path + "sources/dataSet/train_file/eval.txt"
test_path = root_path + "sources/dataSet/train_file/test.txt"
train_eval_path = root_path + "sources/dataSet/train_file/traineval.txt"

log_path = root_path + "src/yolo4/logs/"
weights_path = root_path + "src/yolo4/weights/"

use_gpu = True

fine_tune = False
fine_tune_epoch = 100

batch_size = 8
train_epochs = 500
save_frequency = 2
