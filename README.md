# 深度学习分类项目

<u></u>

### <u>数据集准备</u>

做回转窑气氛分类时拿到的数据集是一些视频文件，因此需要从视频中抽帧提取图片作为数据集。进入python环境运行video_to_imgs.py



### <u>模型训练</u>

需要提前下好预训练模型，这里选择的是resnest50（github上可搜索）,在train.py里可以进行相关参数设置，运行python train.py



### <u>模型预测</u>

在predict.py里指定训练好的模型路径与数据集路径，运行python predict.py



# qinchangwei
# qinchangwei
