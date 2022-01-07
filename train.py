import os
import argparse

"""
1、安装相关依赖
2、将pretrained-model移动至指定位置
"""

import time
import torch
from torch import nn
import torchvision
from torchvision import transforms
from resnest.torch import resnest50


import utils

"""
消除随机因素的影响
"""
torch.manual_seed(2021)


def load_data(dir, is_train):
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if is_train:
        dataset = torchvision.datasets.ImageFolder(
            dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        dataset = torchvision.datasets.ImageFolder(
            dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        sampler = torch.utils.data.SequentialSampler(dataset)

    return dataset, sampler


def initial_model(class_num):
    model =  resnest50(pretrained=True)
    channel_in = model.fc.in_features					# 提取fc层中固定的参数
    model.fc = nn.Linear(channel_in, class_num)				# 修改类别为class_num
    return model


def train_one_epoch(model, criterion, optimizer, data_loader, idx_to_class, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        acc0, acc0_5 = utils.accuracy(output, target, idx_to_class)
        acc = 0.5 * acc0 + 0.5 * acc0_5
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc0'].update(acc0.item(), n=batch_size)
        metric_logger.meters['acc0_5'].update(acc0_5.item(), n=batch_size)
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
    print(' *Train acc@0 {top0.global_avg:.3f} Train acc@0.5 {top5.global_avg:.3f} Train acc@ {top.global_avg:.3f}'
          .format(top0=metric_logger.acc0, top5=metric_logger.acc0_5, top=metric_logger.acc))
    return metric_logger.acc.global_avg,loss

def val_one_epoch(model, criterion, data_loader, idx_to_class, device, print_freq=10):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)  # non_blocking=True指可以并行计算
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            acc0, acc0_5 = utils.accuracy(output, target, idx_to_class)
            acc = 0.5 * acc0 + 0.5 * acc0_5
            #val_acc_list.append(acc0)
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc0'].update(acc0.item(), n=batch_size)
            metric_logger.meters['acc0_5'].update(acc0_5.item(), n=batch_size)
            metric_logger.meters['acc'].update(acc.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' *Val_acc@0 {top0.global_avg:.3f} Val_acc@0.5 {top5.global_avg:.3f} Val acc@ {top.global_avg:.3f}'
          .format(top0=metric_logger.acc0, top5=metric_logger.acc0_5, top=metric_logger.acc))

    return metric_logger.acc.global_avg,loss



def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if \
        torch.cuda.is_available() else torch.device('cpu')

    train_dir = '/home/qinchangwei/xxx/train'
    val_dir = '/home/qinchangwei/xxx/val'

    ckpt_dir = 'ckpt'

    dataset, train_sampler = load_data(train_dir, is_train=True)
    data_loader = torch.utils.data.DataLoader( dataset, batch_size=16, sampler=train_sampler, num_workers=4, pin_memory=True)      # num_workers=4：四个工作进程

    dataset_val, val_sampler = load_data(val_dir, is_train=False)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=16, sampler=val_sampler, num_workers=4, pin_memory=True)

    # provide idx and class relationgships
    idx_to_class = {value: key for key, value in dataset.class_to_idx.items()}

    # get the model using our helper function
    model = initial_model(4)				# 分成4类

    # move model to the right device
    model.to(device)
    criterion = nn.CrossEntropyLoss()			# 集成了损失和激活函数

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-4)			# 模型参数、学习率、动量（它的作用是尽量保持当前梯度的变化方向，一般用0.9）、权重衰减
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)	# torch.optim.lr_scheduler模块提供了一些根据epoch训练次数来调整学习率的方法，StepLR是其中一种
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2)

    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []

    print("Start training")
    for epoch in range(45):		#120
        # train for one epoch, printing every 10 iterations
        train_acc, train_loss = train_one_epoch(model, criterion, optimizer, data_loader, idx_to_class, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

        if val_dir is not None:
            val_acc, val_loss = val_one_epoch(model, criterion, data_loader_val, idx_to_class, device=device)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch}

        if ckpt_dir is not None:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
                print('mkdir: {}'.format(ckpt_dir))
        local_ckpt_path = os.path.join(ckpt_dir, 'model_{}.pth'.format(epoch))

        # write classes.txt
        local_classes_path = os.path.join(ckpt_dir, 'classes.txt')
        with open(local_classes_path, 'w') as f:
            for i in range(len(idx_to_class)):
                f.write(idx_to_class[i] + "\n")

        # save model every 5 steps
        if epoch % 6 == 0:
            utils.save_on_master(checkpoint, local_ckpt_path)
  
    print(train_acc_list)
    print(val_acc_list)
    print(train_loss_list)
    print(val_loss_list)

    with open('record.txt', 'w') as f:
         f.write(str('['))
         for i in range(len(train_acc_list)):   
             f.write(str(train_acc_list[i])+str(','))
         f.write(str('],['))
         for i in range(len(val_acc_list)):        
             f.write(str(val_acc_list[i])+str(','))   
         f.write(str('],['))   
         for i in range(len(train_loss_list)):   
             f.write(str(train_loss_list[i])+str(','))
         f.write(str('],['))
         for i in range(len(val_loss_list)):        
             f.write(str(val_loss_list[i])+str(','))   
         f.write(str(']'))     
   
 
if __name__ == "__main__":
    main()
