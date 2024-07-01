import argparse
import os,sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from os.path import join
from function import *
from datasets import *
from resnet import resnet18, resnet50, resnet101
import sys

batch_size = 40
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
])

def test_acc_AA(net, total_cls, incre_cls, data_source, path): #mometum model
    net.eval()
    acc = []
    for i in range(int(total_cls / incre_cls)):
        confi_cls = [label for label in range(i * incre_cls, (i + 1) * incre_cls)]
        correct = 0
        total = 0

        if data_source.split('/')[0] == 'office-home':
            test_dataset = dataset(dataset=data_source, root=join(path, 'office-home'), noisy_path=None,
                                       mode='all',
                                       transform=transform_test,
                                       incremental=True,
                                       confi_class=confi_cls
                                       )

            test_loader = DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False)

        for idx, data in enumerate(test_loader):
            inputs = data[0].cuda()
            labels = data[2].cuda()
            with torch.no_grad():
                _, outputs = net(inputs)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs, 1)
            total += inputs.size(0)
            num = (predicted == labels).sum()
            correct = correct + num

        acc.append(100. * correct.item() / total)

    return acc

def test_acc_DA(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    for idx, data in enumerate(test_loader):
        inputs = data[0].cuda()
        labels = data[2].cuda()
        with torch.no_grad():
            _, outputs = net(inputs)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.reshape(-1, 1)
        total += inputs.size(0)
        num = (predicted == labels).sum()
        correct = correct + num
    acc = 100. * correct.item() / total

    return acc

def test_buffer_acc(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    for idx, data in enumerate(test_loader):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        with torch.no_grad():
            _, outputs = net(inputs)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.reshape(-1, 1)
        total += inputs.size(0)
        num = (predicted == labels).sum()
        correct = correct + num
    acc = 100. * correct.item() / total

    return acc