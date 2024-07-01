import numpy as np
import os
from OH_source_Train import val_net, Dataset
from OH_datasets import FileListDataset
from os.path import join
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import torch.nn as nn
import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import random
from PIL import ImageFilter
from net.resnet import resnet50
import matplotlib.pyplot as plt
from PIL import Image




class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def plot_hot_matrix(matrix, save_path):
    plt.figure()
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()

def model_perform_valid(net, source_domain, my_dataset,dataset_name = ''):
    net.eval()
    cls_nums = 65
    # cls_nums = 12

    if dataset_name != '':
        dataset_name = dataset_name+'_'

    total_matrix = np.zeros((cls_nums, cls_nums))
    total_matrix_add = np.zeros((cls_nums, cls_nums))
    total_matrix_mul = np.zeros((cls_nums, cls_nums))

    for domain in my_dataset.domains:
        if domain != "Art":
            continue
        # if domain == source_domain:
        #     continue
        # if source_domain != "clipart":
        #     continue
        print('{} {} model in {}'.format(dataset_name,source_domain, domain))
        domain_file = my_dataset.files[my_dataset.domains.index(domain)]
        # domain_classes = [i for i in range(cls_nums)]
        domain_classes = [4,6,8,16,25,28,29,56,63]
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # transform_train = transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.RandomCrop((224, 224)),
        #     transforms.RandomApply(
        #         [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        #         p=0.8,  # not strengthened
        #     ),
        #     transforms.RandomGrayscale(p=0.4),
        #     transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
        # ])
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
        ])

        domain_ds = FileListDataset(list_path=domain_file, path_prefix=my_dataset.prefixes[my_dataset.domains.index(domain)],
                                          transform=transform_train,
                                    # transform= transform_test,
                                          filter=(lambda x: x in domain_classes), return_id=False)
        domain_loader = torch.utils.data.DataLoader(domain_ds, batch_size=1, shuffle=True,
                                                        num_workers=4)
        
        dict_appeared = {}
        for i in domain_classes:
            dict_appeared[i] = 0
        for i, (inputs, labels) in enumerate(domain_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            if dict_appeared[labels.item()]>3:
                continue
            dict_appeared[labels.item()] += 1
            #plot the input and save as fig
            inputs = inputs.squeeze(0)
            inputs = inputs.cpu().numpy()

            inputs = np.transpose(inputs, (1, 2, 0))  
            plt.imshow(inputs)
            plt.axis('off')  # Remove the axis
            plt.tight_layout()
            plt.savefig('input_{}_{}_{}_{}.png'.format(source_domain, domain, labels.item(), dict_appeared[labels.item()]))
            plt.close()
                                              


        # acc = val_net(net, domain_loader,flag=False,cls_num=cls_nums)
        # print("from {} to {}".format(source_domain, domain))
        # print("acc=%.3f" % acc)
    #     # result_root = join(server_root, 'ProCA-main/ProCA-main/results/confusion_matrix/')
    #     result_root = join(server_root, 'ProCA-main/ProCA-main/{}results/confusion_matrix/'.format(dataset_name))

    #     result_root = join(result_root, source_domain)
    #     print(result_root)
    # #
    #     if not os.path.exists(result_root):
    #         os.makedirs(result_root)
    #     add_performance = np.zeros((cls_nums, cls_nums))
    #     mul_performance = np.zeros((cls_nums, cls_nums))
    #     matrix_soft = nn.Softmax(dim=1)(torch.tensor(matrix))

    #     for A_class in range(cls_nums):
    #         for B_class in range(A_class, cls_nums,1):
    #             add_performance[A_class, B_class] = matrix[A_class, B_class] + matrix[B_class, A_class]
    #             # print(matrix[A_class,B_class],matrix[B_class,A_class])
    #             mul_performance[A_class, B_class] = matrix_soft[A_class, B_class] * matrix_soft[B_class, A_class]

    #     total_matrix += matrix
    #     total_matrix_add += add_performance
    #     total_matrix_mul += mul_performance

    #     add_performance = add_performance / add_performance.sum(axis=1, keepdims=True)
    #     mul_performance = mul_performance / mul_performance.sum(axis=1, keepdims=True)
    #     np.save(join(result_root, 'matrix_{}_{}.npy'.format(source_domain, domain)), matrix)
    #     np.save(join(result_root, 'add_performance_{}_{}.npy'.format(source_domain, domain)), add_performance)
    #     np.save(join(result_root, 'mul_performance_{}_{}.npy'.format(source_domain, domain)), mul_performance)

    #     #plot performance
    #     #normalize by row
    #     add_performance = add_performance / add_performance.sum(axis=1, keepdims=True)
    #     mul_performance = mul_performance / mul_performance.sum(axis=1, keepdims=True)
    #     matrix = matrix / matrix.sum(axis=1, keepdims=True)

    #     plot_hot_matrix(add_performance, join(result_root, 'add_performance_{}_{}.png'.format(source_domain, domain)))
    #     plot_hot_matrix(mul_performance, join(result_root, 'mul_performance_{}_{}.png'.format(source_domain, domain)))
    #     plot_hot_matrix(matrix, join(result_root, 'matrix_{}_{}.png'.format(source_domain, domain)))

    # #plot performance
    # #normalize by row
    # add_performance = total_matrix_add / total_matrix_add.sum(axis=1, keepdims=True)
    # mul_performance = total_matrix_mul / total_matrix_mul.sum(axis=1, keepdims=True)
    # total_matrix = total_matrix / total_matrix.sum(axis=1, keepdims=True)

    # plot_hot_matrix(add_performance, join(result_root, 'add_performance_{}_{}.png'.format(source_domain, 'total')))
    # plot_hot_matrix(mul_performance, join(result_root, 'mul_performance_{}_{}.png'.format(source_domain, 'total')))
    # plot_hot_matrix(total_matrix, join(result_root, 'matrix_{}_{}.png'.format(source_domain, 'total')))

    # np.save(join(result_root, 'matrix_{}_{}.npy'.format(source_domain, 'total')), total_matrix)
    # np.save(join(result_root, 'add_performance_{}_{}.npy'.format(source_domain, 'total')), add_performance)
    # np.save(join(result_root, 'mul_performance_{}_{}.npy'.format(source_domain, 'total')), mul_performance)



if __name__ == '__main__':
    my_dataset = Dataset(
        path='./dataset/office-home',
        domains=['Art', 'Clipart', 'Product', 'World'],
        files=[
            'Art.txt',
            'Clipart.txt',
            'Product.txt',
            'World.txt'],
        prefix='./dataset/office-home')
    model_file = ['./model_source/20240414-0211-OH_Art_ce_singe_gpu_resnet50_best_param.pth',
                  './model_source/20240322-1652-OH_Clipart_ce_singe_gpu_resnet50_best_param.pth',
                    './model_source/20240414-0239-OH_Product_ce_singe_gpu_resnet50_best_param.pth',
                  './model_source/20240414-0307-OH_World_ce_singe_gpu_resnet50_best_param.pth']
    server_root = '/public/home/imgbreaker/CIUDA/'

    # my_dataset = Dataset(
    #     path='/public/home/imgbreaker/Desktop/PADA/pytorch/data/office',
    #     domains=['amazon', 'dslr', 'webcam'],
    #     files=[
    #         'amazon_31_list.txt',
    #         'dslr_31_list.txt',
    #         'webcam_31_list.txt'],
    #     prefix='/public/home/imgbreaker/Desktop/PADA/pytorch/dataset/office/domain_adaptation_images')
    # model_file = ['./model_source/office31/20240427-2132-OH_amazon_ce_singe_gpu_resnet50_best.pkl',
    #               './model_source/office31/20240427-2133-OH_dslr_ce_singe_gpu_resnet50_best.pkl',
    #               './model_source/office31/20240427-2211-OH_webcam_ce_singe_gpu_resnet50_best.pkl']
    # cls_nums = 31

    ### DomainNet-126
    # my_dataset = Dataset(
    #     path='/public/home/imgbreaker/Desktop/dataset/DomainNet/data',
    #     domains=['clipart', 'painting', 'real', 'sketch'],
    #     files=[
    #         'clipart_list.txt',
    #         'painting_list.txt',
    #         'real_list.txt',
    #         'sketch_list.txt'
    #     ],
    #     prefix='/public/home/imgbreaker/Desktop/dataset/DomainNet/dataset/')
    # model_file = ['./model_source/domainNet-126/20240429-2246-OH_clipart_ce_singe_gpu_resnet50_best_param.pth',
    #               './model_source/domainNet-126/20240429-2246-OH_painting_ce_singe_gpu_resnet50_best_param.pth',
    #               './model_source/domainNet-126/20240429-2246-OH_real_ce_singe_gpu_resnet50_best_param.pth',
    #               './model_source/domainNet-126/20240429-2246-OH_sketch_ce_singe_gpu_resnet50_best_param.pth']
    # server_root = '/public/home/imgbreaker/CIUDA/'
    # cls_nums = 126

    # my_dataset = Dataset(
    #         path='/public/home/imgbreaker/Desktop/PADA/pytorch/imageCLEF/list/',
    #         domains=['b', 'c', 'i', 'p'],
    #         files=[
    #             'bList.txt',
    #             'cList.txt',
    #             'iList.txt',
    #             'pList.txt'
    #         ],
    #         prefix='/public/home/imgbreaker/Desktop/PADA/pytorch/imageCLEF/')
    # model_file = ['./model_source/imageCLEF/20240502-1058-OH_b_ce_singe_gpu_resnet50_best_param.pth',
    #               './model_source/imageCLEF/20240502-1058-OH_c_ce_singe_gpu_resnet50_best_param.pth',
    #               './model_source/imageCLEF/20240502-1058-OH_i_ce_singe_gpu_resnet50_best_param.pth',
    #               './model_source/imageCLEF/20240502-1058-OH_p_ce_singe_gpu_resnet50_best_param.pth']
    # server_root = '/public/home/imgbreaker/CIUDA/'
    # cls_nums = 12
    cls_nums = 65 
    #set default root path
    for source_domain in my_dataset.domains:
        # if source_domain != 'real':
        #     continue
        if source_domain != 'Art':
            continue
        net = resnet50(pretrained=True)
        # net.fc = nn.Linear(2048, 126)
        net.fc = nn.Linear(2048, cls_nums)
        print(my_dataset.domains.index(source_domain))
        print(model_file[my_dataset.domains.index(source_domain)])
        # net_param = torch.load(model_file[my_dataset.domains.index(source_domain)])
        # net.load_state_dict(net_param)
        net.eval()
        net.cuda()
        dataset_name = ''
        # dataset_name = 'imageCLEF'
        model_perform_valid(net, source_domain, my_dataset,dataset_name)

