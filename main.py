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
                      "Office_31": 10,
                      "DomainNet": 126}
incre_num = {"Office_Home": 10,
             "Office_31": 4,
             "DomainNet": 20}
negative_form = []

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data_dir', type=str, default='dataset')
parser.add_argument('--dataset', default='office-home/Product', type=str)
parser.add_argument('--source', default='Art', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--noisy_path', type=str, default=None)
parser.add_argument('--target', default="Product")

parser.add_argument('--forgetting_eval', default=False)
parser.add_argument('--num_class', default=65, type=int)
parser.add_argument('--temporal_length', default=5, type=int)

parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')

parser.add_argument('--seed', default=123)
parser.add_argument('--deviceid', default=[0,1,2,3])

parser.add_argument('--ctr', action='store_false', help="use contrastive loss")
parser.add_argument('--label_refinement', action='store_false', help="Use label refinement")
parser.add_argument('--neg_l', action='store_true', help="Use negative learning")
parser.add_argument('--reweighting', action='store_true', help="Use reweighting")
parser.add_argument('--pos_l', action='store_true', help="Use positive learning")
parser.add_argument('--source_model', default='./model_source/20240414-0211-OH_Art_ce_singe_gpu_resnet50_best.pkl')
parser.add_argument('--weight_model', default='./model_source/20240416-2141-OH_Art_ce_singe_gpu_resnet50_best_neg_param.pth')

parser.add_argument('--run_name', type=str)
parser.add_argument('--wandb', action='store_true', help="Use wandb")
parser.add_argument('--num_per_time', default="Office_Home")
parser.add_argument('--select_data', default="Office_Home")
parser.add_argument('--txt', default="acc.txt")
parser.add_argument('--model_name', default="normal")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
# CUDA_VISIBLE_DEVICES=gpu_id
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
    neg_num = 10
    batch_size = args.batch_size
    prototypes_update_interval = 8
    # optimizer
    lr = 1e-3
    weight_decay = 1e-6
    momentum = 0.9
    n_epoches = 6
    nll = nn.NLLLoss()

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
    contrastive_label = torch.tensor([0]).cuda()

    current_step = 0
    confi_cls_history = []
    confi_cls_value = np.zeros(total_cls_nums)
    reply_buffer = reply_buffer(transform_test, reply_buffer_nums)

    # pre-trained model
    pretrained_net = resnet50(pretrained=True)
    pretrained_net.fc = nn.Linear(2048, total_cls_nums)
    pretrained_net.load_state_dict(torch.load(args.source_model))
    pretrained_net = pretrained_net.cuda()
    pretrained_net.eval()

    weight_net = resnet50(pretrained=True)
    weight_net.fc = nn.Linear(2048, 2 * total_cls_nums)
    weight_net.load_state_dict(torch.load(args.weight_model))
    weight_net = torch.nn.DataParallel(weight_net, device_ids=args.deviceid)
    weight_net = weight_net.cuda()
    weight_net.eval()

    best_incre_acc = []
    for incre_idx in range(total_cls_nums // 10):    # incre_idx 训练次数
        target_total_classes = [i for i in range(total_cls_nums)]

        target_train_classes = [i for i in range(10 * incre_idx, 10 * incre_idx + 10)]
        target_test_classes = [i for i in range(0, 10 * incre_idx + 10)]
        target_test_forget = [i for i in range(10)]

        if args.dataset.split('/')[0] == 'office-home':
            train_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'office-home'), noisy_path=None,
                                    mode='all',
                                    transform=transform_test,
                                    incremental=True,
                                    confi_class=target_train_classes
                                    )
            if args.forgetting_eval == True:
                test_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'office-home'), noisy_path=None,
                                       mode='all',
                                       transform=transform_test,
                                       incremental=True,
                                       confi_class=target_test_forget
                                       )
            else:
                test_dataset = dataset(dataset=args.dataset, root=join(args.data_dir, 'office-home'), noisy_path=None,
                                       mode='all',
                                       transform=transform_test,
                                       incremental=True,
                                       confi_class=target_test_classes
                                       )

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)

        net = resnet50(pretrained=True)
        net.fc = nn.Linear(2048, total_cls_nums)
        net.load_state_dict(torch.load(args.source_model))
        net = torch.nn.DataParallel(net, device_ids=args.deviceid)
        net = net.cuda()
        if incre_idx == 0:    # 调用原模型
            # net = create_model(arch, args)
            # momentum_net = create_model(arch, args)
            # load_weights(net, 'logs/' + args.source + '/weights_best.tar')
            # load_weights(momentum_net, 'logs/' + args.source + '/weights_best.tar')
            momentum_net = resnet50(pretrained=True)
            momentum_net.fc = nn.Linear(2048, total_cls_nums)
            momentum_net.load_state_dict(torch.load(args.source_model))
            momentum_net = torch.nn.DataParallel(momentum_net, device_ids=args.deviceid)
            momentum_net = momentum_net.cuda()
        else:
            momentum_net = torch.load('./model_source/{}_2_{}_Resnet50_DA_Best_stage{}_{}.pt'.format(source, target, incre_idx - 1, args.model_name))


        moco_model = AdaMoCo(src_model=net, momentum_model=momentum_net, features_length=2048,
                             num_classes=args.num_class, dataset_length=len(train_dataset),
                             temporal_length=args.temporal_length)
        # optimizer
        optimizer = optim.SGD(moco_model.momentum_model.parameters(), lr=args.lr, weight_decay=5e-4)
        # optimizer = optim.Adam(moco_model.momentum_model.parameters(), lr = args.lr, weight_decay=5e-4)


        confi_class_idx, confuse_cls_idx, confi_class_values = get_confi_classes(pretrained_net, train_loader,
                                                                                 total_cls_nums, threshold=0.2)
        print(confi_class_idx)
        print(confi_class_values)
        # pred_label_dict = obtain_label(train_loader, pretrained_net, confi_class_idx)
        # center_dict = get_input_centers(pred_label_dict, moco_model, train_loader)
        pred_label_dict, weight_dict = obtain_label_weight(train_loader, pretrained_net, confi_class_idx,
                                                           total_cls_nums, weight_net)
        center_dict = get_input_centers(pred_label_dict, moco_model, train_loader)

        best_tar_acc = 0.
        final_acc = 0.

        this_stage_save_imgs = True

        for epoch in tqdm(range(n_epoches)):   # 一次task


            if reply_buffer.exemplar_set:  # if there are any prototype-images
                # get the reply buffer loader
                reply_ds = reply_dataset(images=reply_buffer.exemplar_set, labels=confi_cls_history,
                                         buffer_per_class=reply_buffer_nums, soft_predictions=reply_buffer.soft_pred)
                reply_loader = torch.utils.data.DataLoader(reply_ds, batch_size=args.batch_size, shuffle=True)
                iter_reply_buffer = iter(reply_loader)


            sum_contras = torch.tensor(0.).cuda()
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                moco_model.momentum_model.train()
                # momentum_model.train()

                ori_x = batch[0].cuda()
                strong_x = batch[1].cuda()
                y = batch[2].cuda()
                idxs = batch[3].cuda()
                strong_x2 = batch[5].cuda()

                fea, output = moco_model(ori_x, cls_only=False)
                fea = F.normalize(fea)
                fea_strong_x, logit_strong_x, logit_ctr, keys = moco_model(strong_x, strong_x2, cls_only=True)

                # pseudo_labels = []
                # for each_idx in idxs.cpu().numpy().tolist():
                #     pseudo_labels.append(pred_label_dict[each_idx])
                # pseudo_labels = torch.Tensor(pseudo_labels).cuda()
                pseudo_labels = []
                weight = []
                for each_idx in idxs.cpu().numpy().tolist():
                    pseudo_labels.append(pred_label_dict[each_idx])
                    weight.append(weight_dict[each_idx])
                pseudo_labels = torch.Tensor(pseudo_labels).cuda()
                weight = torch.Tensor(weight).cuda()

                loss_ce = (weight * CE(output, pseudo_labels.long())).mean()

                # target entropy loss
                # if args.pos_l:
                #     loss_ce = ce_loss(output, pseudo_labels.long())
                # else:
                #     loss_ce = 0

                # contrastive loss at same epoch
                # if args.ctr:
                #     loss_ctr_cur = contrastive_loss(
                #         logits_ins=logit_ctr,
                #         pseudo_labels=moco_model.mem_labels[idxs],  # current batch past_T labels
                #         mem_labels=moco_model.mem_labels[moco_model.idxs]  # past_T labels of whole dataset
                #     )
                # else:
                #     loss_ctr_cur = 0

                # moco_model.update_memory(epoch, idxs, keys, pseudo_labels, y)  # update current batch label

                # negative learning

                # prob_with_neg = torch.softmax(output, dim=1)
                # if args.neg_l:
                #     # Standard negative learning
                #     loss_neg = (nl_criterion(logit_strong_x, pseudo_labels.long(), confi_class_idx)).mean()
                #     if args.reweighting:
                #         w = reliable_weight(prob_with_neg, total_cls_nums)
                #         loss_neg = (w * nl_criterion(logit_strong_x, pseudo_labels, confi_class_idx)).mean()
                # else:
                #     loss_neg = 0
                #     if args.reweighting:
                #         w = reliable_weight(prob_with_neg, total_cls_nums)
                #         loss_neg = (w * CE(logit_strong_x, pseudo_labels)).mean()

                # buffer
                loss_ctr_bank = torch.tensor(0.).cuda()
                loss_kd = torch.tensor(0.).cuda()
                if reply_buffer.exemplar_set:  # if there are any prototype-images
                    # since the memory limitation is about 100 samples, we repeat 100/batch size times to calculate all samples
                    data_buffer = next(iter_reply_buffer, -1)
                    if data_buffer == -1:
                        data_target_iter = iter(reply_loader)
                        re_org_img, re_org_label, re_org_sp = data_target_iter.next()
                    else:
                        re_org_img, re_org_label, re_org_sp = data_buffer

                    re_org_img = re_org_img.cuda()
                    re_org_label = re_org_label.cuda()
                    re_org_sp = re_org_sp.cuda()

                    reply_feas, reply_ouputs = moco_model(re_org_img)
                    # print("fea", fea)
                    # print("reply_feas", reply_feas)
                    buffer_center = get_buffer_centers(reply_buffer, moco_model, neg_num)
                    buffer_center = buffer_center.cuda()
                    buffer_center = F.normalize(buffer_center)
                    # contrastive loss with bank

                    for i in range(len(fea)):
                        pseudo = pseudo_labels[i].cpu().numpy().tolist()
                        pseudo = int(pseudo)
                        center_fea = center_dict[pseudo].cuda()
                        numerator = torch.exp(torch.dot(fea[i], center_fea) / gamma)

                        deno = torch.matmul(buffer_center, fea[i]).div(gamma)
                        deno = torch.sum(torch.exp(deno))
                        loss_each = - torch.log(numerator / (deno + numerator))
                        loss_ctr_bank += loss_each

                    loss_ctr_bank = loss_ctr_bank / len(fea)
                    # loss_ctr_bank.requires_grad_(True)
                    #########################################################
                    # distillation
                    # calculate the sf-pred cross-entropy
                    reply_ouputs = nn.Softmax(dim=1)(reply_ouputs)  # get the softmax-output
                    reply_ouputs = torch.log(reply_ouputs)  # get the log-softmax
                    soft_pred_loss = torch.sum(-1 * re_org_sp * reply_ouputs, dim=1)  # -1 * p(x) * log q(x)
                    soft_pred_loss = torch.mean(soft_pred_loss)

                    loss_kd += soft_pred_loss

                total_loss = loss_ce + loss_ctr_bank * 0.1 + loss_kd
                total_loss.requires_grad_(True)
                # print("loss_kd", loss_kd)
                # print("loss_ce", loss_ce)
                # print("loss_neg", loss_neg)
                # print("loss_ctr_bank", loss_ctr_bank)
                # print("loss_ctr_cur", loss_ctr_cur)
                # print("total_loss", total_loss)
                # print(" ")
                total_loss.backward()
                optimizer.step()

            if epoch == 3 or (
                    epoch != 0 and epoch % prototypes_update_interval == 0):  # after the warm-up stage, update the imgs 1 time / per 3 epoches
                for confi_idx, confi_class in enumerate(confi_class_idx):
                    if this_stage_save_imgs:
                        if confi_class not in confi_cls_history:
                            confi_cls_history.append(confi_class)
                            confi_cls_value[confi_class] = confi_class_values[confi_idx]

                            imgs = get_one_classes_imgs(train_loader, confi_class, pred_label_dict)
                            reply_buffer.construct_exemplar_set(imgs, moco_model.momentum_model)
                    else:
                        history_idx = confi_cls_history.index(confi_class)
                        if confi_class_values[confi_idx] >= confi_cls_value[confi_class]:  # 大于或者等于都更新
                            imgs = get_one_classes_imgs(train_loader, confi_class, pred_label_dict)
                            reply_buffer.update_exemplar_set(imgs, moco_model.momentum_model, history_idx)
                this_stage_save_imgs = False

            # acc = test_acc_AA(momentum_net, (incre_idx+1) * incre_cls_nums, incre_cls_nums, args.dataset, args.data_dir)
            # total_mean_acc = test_acc_DA(moco_model.momentum_model, test_loader)
            # print('total_mean_acc is %.3f' % total_mean_acc)

            total_mean_acc = test_acc_DA(moco_model.momentum_model, train_loader)
            print('total_mean_acc is %.3f' % total_mean_acc)
            with open(args.txt, 'a') as f:
                f.write('epoch is %.3f\n' % epoch)
                f.write('total_mean_acc is %.3f\n' % total_mean_acc)

            if epoch == 0:
                best_tar_acc = total_mean_acc
                torch.save(moco_model.momentum_model,
                           './model_source/{}_2_{}_Resnet50_DA_Best_stage{}_{}.pt'.format(source, target, incre_idx,
                                                                                          args.model_name))
            else:
                if total_mean_acc > best_tar_acc:
                    best_tar_acc = total_mean_acc
                    torch.save(moco_model.momentum_model,
                               './model_source/{}_2_{}_Resnet50_DA_Best_stage{}_{}.pt'.format(source, target, incre_idx,
                                                                                              args.model_name))
            if incre_idx == total_cls_nums // 10 - 1:
                fin_acc = test_acc_DA(moco_model.momentum_model, test_loader)
                final_acc = max(final_acc, fin_acc)
                with open(args.txt, 'a') as f:
                    f.write('\n final acc is %.3f\n' % fin_acc)
        best_incre_acc.append(best_tar_acc)

    print("final_acc", final_acc)
    print("best_incre_acc", best_incre_acc)
    with open(args.txt, 'a') as f:
        f.write('final_acc is %.3f\n' % final_acc)
        for i in range(len(best_incre_acc)):
            f.write('i-th task acc is %.5f\n' % best_incre_acc[i])


