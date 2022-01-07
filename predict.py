import torch
import os
from PIL import Image
import pandas as pd         #基于NUMPY的一种工具
from tqdm import tqdm		#可扩展的进度条
import numpy as np
from collections import Counter		#Couter是一个字典子类
from transform import tta_test_transform
from resnest.torch import resnest50
from torch import nn
import torch.nn.functional as F
from video_to_imgs import trans


def initial_model(class_num):
    model =  resnest50(pretrained=False)
    channel_in = model.fc.in_features					# 提取fc层中固定的参数
    model.fc = nn.Linear(channel_in, class_num)				# 修改类别为class_num
    return model

def load_checkpoint(filepath):
    model = initial_model(4)
    device = torch.device('cuda') if \
                   torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        model.to(device)
        model.load_state_dict(torch.load(filepath)['model'])
    else:
        model.load_state_dict(torch.load(filepath,
                                             map_location='cpu')['model'])
    #model = torch.load(filepath)                     # 用来加载torch.save()保存的模型文件
    #for parameter in model.parameters():             # 取出模型参数
        #parameter.requires_grad = False              # 固定特征层的参数，不会发生梯度的更新
    model.eval()             # 固定特征层的参数，不会发生梯度的更新				     # 评估模式，batchNorm,dropout层等用于优化训练而添加的网络层会被关闭，参数不发生变化
    return model




def tta_predict(model):
    # 读入模型
    model = load_checkpoint(model)
    print('..... Finished loading model! ......')
    ##将模型放置在gpu上运行
    device = torch.device('cuda') if \
                   torch.cuda.is_available() else torch.device('cpu')
    #if torch.cuda.is_available():
        #model.to(device)
    pred_list,conf_list = [], []
    
    imgs = os.listdir(img_dir)

    for i in tqdm(range(len(imgs))):
        
        img_path = imgs[i].strip().split(',')[0]
        img = os.path.join(img_dir,img_path)
        img1 = Image.open(img).convert('RGB')
        # print(type(img))
        pred,confs = [],[]
        for i in range(8):
            img = tta_test_transform(size=224)(img1).unsqueeze(0)

            if torch.cuda.is_available():
                img = img.to(device)							     # 转移到CUDA进行处理
            with torch.no_grad():                             # 不会track梯度，使得新增的tensor没有梯度，带梯度的tensor能够进行原地运算
                out = model(img)			      # out为一个tensor变量
            res_softmax = F.softmax(out[0],dim=0).to('cpu')   # dim=0时，按列softmax,列和为1
            conf, prediction = res_softmax.topk(1, 0, True, True)  # 返回tensor中某个dim的前k个数据及其对应的index。dim=0时返回某一列的最大值和行号
            
            #prediction = torch.argmax(out, dim=1).cpu().item()    #返回指定维度最大值的序号，dim=1时为求每一行的最大列标号，dim=0时为求每一列的最大行标号
            pred.append(prediction)
            confs.append(conf)
            #res.append(res_softmax)
        res = Counter(pred).most_common(1)[0][0]   #返回序列中出现最多的元素，most_common(1)返回一个二元元组，元组第一个值是元素的值，第二个值是该元素的出现次数
        score = Counter(confs).most_common(1)[0][0] 
        #confs = torch.Tensor(confs)
        #x = confs.topk(1, 0, True, True)
        #x = x.numpy()[0]
        #y = res[x]

        pre_label = labels[res.numpy()[0]]

        pred_list.append(pre_label)
        conf_list.append(score.numpy()[0])

        most_label = Counter(pred_list).most_common(1)[0][0]

    return most_label

if __name__ == "__main__":

    trained_model = '/home/qinchangwei/xxx/ckpt/model_42.pth'
    model_name = 'resnest50'
    """
    with open('test.csv',  'r')as f:
        imgs = f.readlines()
    """
   
    video_path = '/home/qinchangwei/xxx/气氛差1'+str('.mp4') 

    labels = []
    #img_dir = '/home/qinchangwei/xxx/predict'
    img_dir = trans(video_path)
    class_path ='classes.txt'
    with open(class_path) as lines:
         for line in lines:
             line = line.strip()
             labels.append(line)

    predict_label = tta_predict(trained_model)

    print("预测：")
    print(predict_label)

    if predict_label == 1:
       print("不清晰！")

    if predict_label == 2:
       print("模糊！")

    if predict_label == 3:
       print("气氛差！")

    if predict_label == 4:
       print("气氛清晰！")
