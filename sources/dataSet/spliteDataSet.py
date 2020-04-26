# =============================================
# -*- coding: utf-8 -*-           
# @Time    : 2020/4/26 7:59 下午    
# @Author  : xiao9616           
# @Email   : 749935253@qq.com   
# @File    : spliteDataSet.py
# @Software: PyCharm
# ============================================
# 划分train/eval/test数据集（文件名）
import os
import random

xmlPath = './Annotations/xml'
savePath = './train_eval_test'

train_eval_percent = 0.9
train_percent = 0.8

temp_xml = os.listdir(xmlPath)
anno_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        anno_xml.append(xml)

num_total = len(anno_xml)
num_list = range(num_total)
train_eval_num = random.sample(num_list, int(num_total * train_eval_percent))
train_num = random.sample(train_eval_num, int(num_total * train_eval_percent * train_percent))

train_eval_file = open(os.path.join(savePath, "traineval.txt"), 'w')
train_file = open(os.path.join(savePath, "train.txt"), 'w')
eval_file = open(os.path.join(savePath, "eval.txt"), 'w')
test_file = open(os.path.join(savePath, "test.txt"), 'w')

for i in num_list:
    name = anno_xml[i][:-4] + '\n'
    if i in train_eval_num:
        train_eval_file.write(name)
        if i in train_num:
            train_file.write(name)
        else:
            eval_file.write(name)
    else:
        test_file.write(name)

train_eval_file.close()
train_file.close()
eval_file.close()
test_file.close()

print("splite dataset successful ")
