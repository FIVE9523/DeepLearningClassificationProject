import os
import shutil
import random

path1 = '/home/qinchangwei/xxx/img'
path2 = '/home/qinchangwei/xxx/train'
valid_size = 0.8
path_1_doc = os.listdir(path1)
path_1_doc_path = [path1 + '/' + i for i in path_1_doc]

for doc in path_1_doc:  # 新建空文件夹，使得和path1里的一样
    if not os.path.exists(path2 + '/' + doc):
        os.mkdir(path2 + '/' + doc)
path_2_doc = os.listdir(path2)

for k in range(len(path_1_doc)):
    i = path1 + '/' + path_1_doc[k]
    image_list = os.listdir(i)
    image_detailed = [i + '/' + j for j in image_list]
    img_num = len(image_list)
    random_index = random.sample(range(0, img_num - 1), int(valid_size * img_num))
    for i in random_index:
        shutil.move(image_detailed[i], path2 + '/' + path_1_doc[k])

print("移动完成")
