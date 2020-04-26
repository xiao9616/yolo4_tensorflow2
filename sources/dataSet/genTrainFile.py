# =============================================
# -*- coding: utf-8 -*-           
# @Time    : 2020/4/26 9:15 下午    
# @Author  : xiao9616           
# @Email   : 749935253@qq.com   
# @File    : genTrainFile.py         
# @Software: PyCharm
# ============================================

import xml.etree.ElementTree as et

set = ['train.txt', 'eval.txt', 'traineval.txt', 'test.txt']
classes = []
xmlPath = './Annotations/xml/'
jpegPath='/Users/xuan/git/yolo4_tensorflow2/sources/dataSet/JPEGImages/'
savePath = './train_file/'
classPath = './Annotations/classes.txt'
train_eval_test = './train_eval_test/'


def parseXML(imageId,listFile):
    xmlfile=open(xmlPath+imageId+'.xml')
    tree=et.parse(xmlfile)
    root=tree.getroot()
    listFile.write(jpegPath+imageId+'.jpg')

    for obj in root.iter('object'):
        difficult=obj.find('difficult').text
        cls=obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_index=classes.index(cls)
        xmlbox=obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        listFile.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_index))
    listFile.write('\n')

if __name__ == '__main__':

    classFile = open(classPath, 'r')

    for line in classFile.readlines():
        line = line.strip('\n')
        classes.append(line)

    for i in set:
        imageIdList=open(train_eval_test+i).read().strip().split()
        listFile=open(savePath+i,'w')
        for imageId in imageIdList:
            parseXML(imageId,listFile)
        listFile.close()

print("generate train file successful ")