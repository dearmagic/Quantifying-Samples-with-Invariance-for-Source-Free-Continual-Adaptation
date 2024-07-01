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
import random, pdb, math, copy_main
from tqdm import tqdm
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from PIL import Image
from moco import *
from os.path import join
from function import *
from datasets import *
from resnet import resnet18, resnet50, resnet101
from test_acc import *
import sys
import gc


"""
    每个round 都要safe模型
    moco_model 在跑完一个task重置
    office-home office-31 domainnet
    dataset:
            origin
            filter
            augmentation
"""

dataset_categories = {"Office_Home": 65,
                      "Office_31": 31,
                      "DomainNet": 126}
incre_num = {"Office_Home": 10,
             "Office_31": 10,
             "DomainNet": 20}
negative_form = []

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data_dir', type=str, default='dataset')
parser.add_argument('--dataset', default='office-31/webcam', type=str)
parser.add_argument('--source', default='amazon', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--noisy_path', type=str, default=None)
parser.add_argument('--target', default="dslr")

parser.add_argument('--forgetting_eval', default=False)
parser.add_argument('--num_class', default=65, type=int)
parser.add_argument('--temporal_length', default=5, type=int)

parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')

parser.add_argument('--seed', default=123)
parser.add_argument('--deviceid', default=[0])

parser.add_argument('--ctr', action='store_false', help="use contrastive loss")
parser.add_argument('--label_refinement', action='store_false', help="Use label refinement")
parser.add_argument('--neg_l', action='store_true', help="Use negative learning")
parser.add_argument('--reweighting', action='store_true', help="Use reweighting")
parser.add_argument('--pos_l', action='store_true', help="Use positive learning")
parser.add_argument('--source_model', default='./model_source/20240427-2132-OH_amazon_ce_singe_gpu_resnet50_best.pkl')
# parser.add_argument('--source_model', default='./model_source/Art_2_Product_Resnet50_DA_Best_stage3_final_buffer_pos_4_24.pt')

parser.add_argument('--run_name', type=str)
parser.add_argument('--wandb', action='store_true', help="Use wandb")
parser.add_argument('--num_per_time', default="Office_31")
parser.add_argument('--select_data', default="Office_31")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
gc.collect()
torch.cuda.empty_cache()
CE = nn.CrossEntropyLoss(reduction='none')  # 算每一个样本的标准差
CEloss = nn.CrossEntropyLoss()

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
])

# transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
#                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

if __name__ == '__main__':

    # writer = SummaryWriter()

    last_acc_stages = []
    best_acc_stages = []
    center_acc_stages = []
    top_ten_stages = []

    # training in the clipart domain
    source = args.source
    target = args.target
    total_cls_nums = dataset_categories[args.select_data]
    incre_cls_nums = incre_num[args.num_per_time]
    reply_buffer_nums = 10  # 10
    batch_size = args.batch_size
    prototypes_update_interval = 15
    # optimizer
    lr = 1e-3
    weight_decay = 1e-6
    momentum = 0.9
    n_epoches = 30

    # # dataset
    # my_dataset = Dataset(
    #     path='./dataset/office-home',
    #     domains=['Art', 'Clipart', 'Product', 'Real_World'],
    #     files=[
    #         'Art.txt',
    #         'Clipart.txt',
    #         'Product.txt',
    #         'World.txt'
    #     ],
    #     prefix='./dataset/office-home')
    # source_file = my_dataset.files[source]  # 路径
    # target_file = my_dataset.files[target]

    # 改变图片格式
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
    ])
    # loss functions
    margin = 0.3
    gamma = 0.07
    nll = nn.NLLLoss()  # https://blog.csdn.net/Jeremy_lf/article/details/102725285 softmax + log + nll = 交叉熵
    ce_loss = nn.CrossEntropyLoss()


    best_incre_acc = []
    target_test_forget = []
    target_test_classes = [i for i in range(30)]
    if args.dataset.split('/')[0] == 'office-home':
        test_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'office-home'), noisy_path=None,
                               mode='all',
                               transform=transform_test,
                               incremental=True,
                               confi_class=target_test_classes
                               )
    elif args.dataset.split('/')[0] == 'office-31':
        test_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'office-31'), noisy_path=None,
                               mode='all',
                               transform=transform_test,
                               incremental=True,
                               confi_class=target_test_classes
                               )

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    net = resnet50(pretrained=True)
    net.fc = nn.Linear(2048, total_cls_nums)
    net.load_state_dict(torch.load(args.source_model, map_location='cpu'))
    net = torch.nn.DataParallel(net, device_ids=args.deviceid)
    net = net.cuda()
    net.eval()


    # net = torch.load(args.source_model)

    acc = test_acc_DA(net, test_loader)
    print(acc)

#best_incre_acc [77.57847533632287, 80.34251675353686, 78.71930669234473, 77.19361856417694, 73.77423033067275, 71.21540312876053]
