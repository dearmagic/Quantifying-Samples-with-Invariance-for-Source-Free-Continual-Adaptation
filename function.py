import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from PIL import Image
from moco import *
import random

def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p+1e-5), dim=axis)


def cosine_similarity(feature, pairs):
    feature = F.normalize(feature)
    pairs = F.normalize(pairs)
    similarity = feature.mm(pairs.t())
    return similarity

def contrastive_loss(logits_ins, pseudo_labels, mem_labels):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    mask = torch.ones_like(logits_ins, dtype=torch.bool)
    mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1) != mem_labels.unsqueeze(0), dim=2)
    logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    return loss


def nl_criterion(output, y, confi_class):
    output_confi = output.clone()
    # _, cls = output_confi.shape
    # for i in range(cls):
    #     if i not in confi_class:
    #         output_confi[:, i] = float('-inf')
    prob = F.softmax(output_confi, dim=1)
    prob_neg = torch.log(torch.clamp(1. - prob, min=1e-5, max=1.))
    ran = []
    for i in range(len(y)):
        _, idx = torch.topk(prob_neg[i], 2)
        if idx[0] == y[i]:
            ran.append(idx[-1])
        else:
            ran.append(idx[0])
    ran = torch.LongTensor(ran).cuda()
    # labels_neg = ((y.unsqueeze(-1).repeat(1, 1) + torch.LongTensor(len(y), 1).random_(1, num_class).cuda()) % num_class).view(-1)

    l = F.nll_loss(prob_neg, ran, reduction='none')

    return l


# def generate_pseudo_prob(output, negative_form, method = "split"):
#     if method == "split":
#         prob = nn.Softmax(dim=1)(output)
#         for i in prob:
#             for j in range(len(negative_form)):
#                 l1, l2 = negative_form[j]
#                 i[l1] += i[class_num + j] * i[l1] / (i[l1] + i[l2])
#                 i[l2] += i[class_num + j] * i[l2] / (i[l1] + i[l2])
#
#     if method == "abandon":
#         prob = nn.Softmax(dim=1)(output[:, 0:class_num])
#
#     return prob

def get_one_classes_imgs(target_train_loader, confi_class_idx, confi_label_dict):
    start_test = True
    with torch.no_grad():
        for _, data in enumerate(target_train_loader):
            inputs = data[0]
            labels = data[2]
            sample_idx = data[3]

            if start_test:
                all_inputs = inputs.float().cpu()
                all_idx = sample_idx.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_inputs = torch.cat((all_inputs, inputs.float().cpu()), 0)
                all_idx = torch.cat((all_idx, sample_idx.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

        # print('construct class %s examplar.' % (class_idx))
        class_img = {}
        img_idx = {}
        for class_idx in confi_class_idx:
            imgs_idx = []
            for cnt_idx, idx in enumerate(all_idx):
                if int(idx.item()) in confi_label_dict:
                    if confi_label_dict[int(idx.item())] == class_idx:
                        imgs_idx.append(cnt_idx)
            class_img[class_idx], img_idx[class_idx] = all_inputs[imgs_idx], all_idx[imgs_idx]

        return class_img, img_idx

def obtain_weight(loader, num_class, weight_net):
    weight_net.eval() # net-M
    start_test = True
    import time
    with torch.no_grad():
        start = time.time()
        for batch_idx, data in enumerate(loader):
            inputs = data[0]
            labels = data[2]
            idx = data[3]
            inputs = inputs.cuda()
            state1_time = time.time()
            _, w_output = weight_net(inputs)
            state2_time = time.time()
            if start_test:
                all_w_output = w_output.float().cpu()
                all_idx = idx
                start_test = False
            else:
                all_w_output = torch.cat((all_w_output, w_output.float().cpu()), 0)
                all_idx = torch.cat((all_idx, idx), 0)
            state3_time = time.time()
    all_w_output = nn.Softmax(dim=1)(all_w_output)
    w = reliable_weight(all_w_output, num_class)

    return dict(zip(all_idx.numpy().tolist(), w.numpy().tolist()))

def obtain_label(loader, net, confi_class_idx):
    net.eval() # net-M
    start_test = True
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            inputs = data[0]
            labels = data[2]
            idx = data[3]
            inputs = inputs.cuda()
            feas, outputs = net(inputs)  # feas ? G(sample)?

            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_idx = idx
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_idx = torch.cat((all_idx, idx), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    # all_output = generate_pseudo_prob(all_output, negative_form)

    _, predict = torch.max(all_output, 1)   # 0,1,2，...
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    # soft label calculate center
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)  # || xTx || ？ equation (3)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])   # 算shared class中心

    # c_0 only get the confident-classes
    initc = initc[confi_class_idx]

    dd = cdist(all_fea, initc, 'cosine')  # 计算两个集合的距离
    pred_label = dd.argmin(axis=1).tolist()  # Eq（4） sample选择最靠近的中心  0,1,2
    confi_class_idx = np.array(confi_class_idx)
    prediction_c0 = confi_class_idx[pred_label]
    # change to real label index
    # acc = np.sum(prediction_c0 == all_label.float().numpy()) / len(all_fea)

    # calculate c1 and pseudo-labels
    # hard label calculate center
    K = all_output.size(1)
    for round in range(1):
        aff = np.eye(K)[prediction_c0]  # 转成one-hot编码

        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        # only get the confident-classes
        initc = initc[confi_class_idx]

        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        prediction_c1 = confi_class_idx[pred_label]
        # acc = np.sum(prediction_c1 == all_label.float().numpy()) / len(all_fea)

    # log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    # print(log_str + '\n')
    # print(all_idx)
    return dict(zip(all_idx.numpy().tolist(), prediction_c1))  # , accuracy * 100, acc * 100
    #       (idx，label)

def obtain_label_weight(loader, net, confi_class_idx, num_class, weight_net):
    net.eval() # net-M
    start_test = True
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            inputs = data[0]
            labels = data[2]
            idx = data[3]
            inputs = inputs.cuda()
            feas, outputs = net(inputs)
            _, w_output = weight_net(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_w_output = w_output.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_idx = idx
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_w_output = torch.cat((all_w_output, w_output.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_idx = torch.cat((all_idx, idx), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    all_w_output = nn.Softmax(dim=1)(all_w_output)

    w = reliable_weight(all_w_output, num_class)
    _, predict = torch.max(all_output, 1)


    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

    initc = initc[confi_class_idx]

    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1).tolist()
    confi_class_idx = np.array(confi_class_idx)
    prediction_c0 = confi_class_idx[pred_label]

    K = all_output.size(1)
    for round in range(1):
        aff = np.eye(K)[prediction_c0]

        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        initc = initc[confi_class_idx]

        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        prediction_c1 = confi_class_idx[pred_label]


    return dict(zip(all_idx.numpy().tolist(), prediction_c1)), dict(zip(all_idx.numpy().tolist(), w.numpy().tolist()))

# Shared class detection
def get_confi_classes(source_model, target_data_loader, class_num, threshold=0.2):
    source_model.eval()
    prediction_bank = torch.zeros(1, class_num).cuda()
    for j, (img_data, _, _, _, _, _) in enumerate(target_data_loader):
        img_data = img_data.cuda()
        with torch.no_grad():
            _, output = source_model(img_data)
        output_prob = F.softmax(output, dim=1).data
        output_prob = output_prob[:, 0:class_num]
        batch_prob_sum = torch.sum(output_prob, dim=0)
        prediction_bank += batch_prob_sum

    confi_class_idx = []
    sort_bank, sort_class_idx = torch.sort(prediction_bank, descending=True)

    # min max scaler 还有一步归一化
    sort_bank = sort_bank.squeeze(0)
    prediction_bank = prediction_bank.squeeze(0)
    max_cls = sort_bank[0]
    min_cls = sort_bank[-1]
    for idx, value in enumerate(prediction_bank):
        prediction_bank[idx] = (prediction_bank[idx] - min_cls) / (max_cls - min_cls)

    avg_prob = torch.mean(prediction_bank)
    confuse_cls_idx = []

    for idx, value in enumerate(prediction_bank):
        if value >= threshold:
            confi_class_idx.append(idx)

        elif threshold > value > avg_prob:
            confuse_cls_idx.append(idx)

    return confi_class_idx, confuse_cls_idx, prediction_bank[confi_class_idx]
    # 返回确定的，模糊的class index，各个label的cumulative prob

def get_confi_classes_with_weight(source_model, target_data_loader, class_num, weight_dict, threshold=0.2):
    source_model.eval()
    prediction_bank = torch.zeros(1, class_num).cuda()
    for j, (img_data, _, _, idx, _, _) in enumerate(target_data_loader):
        img_data = img_data.cuda()
        idx = idx.cuda()
        weight = []
        with torch.no_grad():
            for each_idx in idx.cpu().numpy().tolist():
                weight.append(weight_dict[each_idx])
            weight = torch.Tensor(weight).cuda()
            _, output = source_model(img_data)
            output_prob = F.softmax(output, dim=1).data
            output_prob = output_prob[:, 0:class_num]
            output_prob = (output_prob == output_prob.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
        output_prob = weight * output_prob
        batch_prob_sum = torch.sum(output_prob, dim=0)
        prediction_bank += batch_prob_sum

    confi_class_idx = []
    # sort_bank, sort_class_idx = torch.sort(prediction_bank, descending=True)

    # min max scaler 还有一步归一化
    # sort_bank = sort_bank.squeeze(0)
    prediction_bank = prediction_bank.squeeze(0)
    # max_cls = sort_bank[0]
    # min_cls = sort_bank[-1]
    # for idx, value in enumerate(prediction_bank):
    #     prediction_bank[idx] = (prediction_bank[idx] - min_cls) / (max_cls - min_cls)

    avg_prob = torch.mean(prediction_bank)
    confuse_cls_idx = []

    for idx, value in enumerate(prediction_bank):
        if value >= threshold:
            confi_class_idx.append(idx)

        elif threshold > value > avg_prob:
            confuse_cls_idx.append(idx)

    return confi_class_idx, confuse_cls_idx, prediction_bank[confi_class_idx]

def reliable_weight(prob, class_num):
    negative_all = torch.sum(prob[:, class_num:], dim=1).reshape(-1, 1)
    neg_entropy = - negative_all * torch.log2(1 - negative_all)
    # new_prob = torch.cat(prob[:, 0:class_num], negative_all, 1)
    new_prob = prob[:, 0:class_num]
    max_entropy = torch.log2(torch.tensor(class_num))
    new_prob = entropy(new_prob).reshape(-1, 1) / max_entropy
    w_uncertain = torch.sum(torch.cat((new_prob, neg_entropy), dim=1), dim=1)
    w = torch.exp(-w_uncertain)
    return w.reshape(-1,1)

def compute_class_mean(model, images):
    model.eval()
    with torch.no_grad():
        x = images.cuda()
        feas, output = model(x)
        feature_extractor_output = F.normalize(feas.detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)

        class_center = np.mean(feas.detach().cpu().numpy(), axis=0)

        # get the probability
        output = nn.Softmax(dim=1)(output)

    return class_mean, feature_extractor_output, output, class_center  # output soft_label

class reply_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, buffer_per_class, soft_predictions, weights):
        super(reply_dataset, self).__init__()

        start_cat = True
        for imgs in images:
            for img in imgs:
                if start_cat:
                    img = img.clone().detach()
                    self.images = img.unsqueeze(0).cuda()

                    start_cat = False
                else:
                    img = img.clone().detach()
                    self.images = torch.cat((self.images, img.unsqueeze(0).cuda()), dim=0)

        start_cat = True
        for label in labels:
            for i in range(buffer_per_class):
                if start_cat:
                    self.labels = torch.tensor(label).unsqueeze(0)
                    start_cat = False
                else:
                    self.labels = torch.cat((self.labels, torch.tensor(label).unsqueeze(0)))

        start_cat = True
        for soft_preds in soft_predictions:
            for soft_pred in soft_preds:
                if start_cat:
                    self.batch_soft_pred = soft_pred.unsqueeze(0).cuda()
                    start_cat = False
                else:
                    self.batch_soft_pred = torch.cat((self.batch_soft_pred.cuda(), soft_pred.unsqueeze(0).cuda()), dim=0)

        start_cat = True
        for weight in weights:
            for w in weight:
                if start_cat:
                    self.weights = w.unsqueeze(0).cuda()
                    start_cat = False
                else:
                    self.weights = torch.cat((self.weights, w.unsqueeze(0).cuda()), dim=0)

        self.images = self.images.cpu()
        self.labels = self.labels.cpu()
        self.batch_soft_pred = self.batch_soft_pred.cpu()
        self.weights = self.weights.cpu()

    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.batch_soft_pred[index], self.weights[index]

    def __len__(self):
        return self.labels.shape[0]



class reply_buffer():
    def __init__(self, transform, imgs_per_class=20):
        super(reply_buffer, self).__init__()
        self.exemplar_set = []
        self.soft_pred = []
        self.target_center_set = []
        self.transform = transform
        self.m = imgs_per_class
        self.weight_set = []

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def construct_exemplar_set(self, images, model):  # Eq(5)
        class_mean, feature_extractor_output, buffer_output, class_center = compute_class_mean(model, images)
        exemplar = []
        soft_predar = []
        feas_past = []
        now_class_mean = np.zeros((1, 2048))  # for ResNet-50

        for i in range(self.m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]

            exemplar.append(images[index])   # image prototype
            soft_predar.append(buffer_output[index].unsqueeze(0)) # soft-label
            feas_past.append(feature_extractor_output[index])

        self.exemplar_set.append(exemplar)
        self.soft_pred.append(soft_predar)

    def update_exemplar_set(self, images, model, history_idx):
        class_mean, feature_extractor_output, buffer_output, class_center = compute_class_mean(model, images)
        exemplar = []
        soft_predar = []
        feas_past = []
        now_class_mean = np.zeros((1, 2048))  # for ResNet-50

        for i in range(self.m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])
            soft_predar.append(buffer_output[index].unsqueeze(0))
            feas_past.append(feature_extractor_output[index])

        self.exemplar_set[history_idx] = exemplar
        self.soft_pred[history_idx] = soft_predar

    def construct_exemplar_set_weight(self, images, idx, weight_dict, model):  # Eq(5)
        class_mean, feature_extractor_output, buffer_output, class_center = compute_class_mean(model, images)
        exemplar = []
        soft_predar = []
        feas_past = []
        weight_all = []
        for i in idx.cpu().numpy().tolist():
            weight_all.append(weight_dict[i])
        weight_all = torch.Tensor(weight_all).cuda()
        weight = []
        now_class_mean = np.zeros((1, 2048))  # for ResNet-50

        for i in range(self.m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]

            exemplar.append(images[index])  # image prototype
            soft_predar.append(buffer_output[index].unsqueeze(0))  # soft-label
            feas_past.append(feature_extractor_output[index])
            weight.append(weight_all[index])
            
        self.exemplar_set.append(exemplar)
        self.soft_pred.append(soft_predar)
        self.weight_set.append(weight)

    def update_exemplar_set_weight(self, images, idx, weight_dict, model, history_idx):
        class_mean, feature_extractor_output, buffer_output, class_center = compute_class_mean(model, images)
        exemplar = []
        soft_predar = []
        feas_past = []
        weight_all = []
        weight = []
        for i in idx.cpu().numpy().tolist():
            weight_all.append(weight_dict[i])
        weight_all = torch.Tensor(weight_all).cuda()

        now_class_mean = np.zeros((1, 2048))  # for ResNet-50

        for i in range(self.m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])
            soft_predar.append(buffer_output[index].unsqueeze(0))
            feas_past.append(feature_extractor_output[index])
            weight.append(weight_all[index])

        self.exemplar_set[history_idx] = exemplar
        self.soft_pred[history_idx] = soft_predar
        self.weight_set[history_idx] = weight

def get_buffer_centers(reply_buffer, net, neg_num):
    net.eval()
    buffer_center = []
    negative_center = []
    for i in reply_buffer.exemplar_set:
        i_tensor = torch.stack(i, dim=0)
        _, _, _, i_center = compute_class_mean(net, i_tensor)
        buffer_center.append(i_center)
    x = [i for i in range(len(buffer_center))]
    for i in range(neg_num):
        negative_center.append(buffer_center[random.choice(x)])
    negative_center = np.array(negative_center)
    return torch.Tensor(negative_center)

def get_input_centers(pesudo_label, net, loader):
    net.eval()
    start_test = True
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            inputs = data[0]
            idx = data[3]
            idx_tmp = idx.numpy().tolist()
            hard_label = []
            for i in idx_tmp:
                hard_label.append(pesudo_label[i])
            hard_label = torch.Tensor(hard_label)
            inputs = inputs.cuda()
            feas, outputs = net(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_idx = idx.float()
                all_label = hard_label
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_idx = torch.cat((all_idx, idx.float()), 0)
                all_label = torch.cat((all_label, hard_label), 0)
        all_fea = F.normalize(all_fea.detach()).cpu()

    center_dict = {}
    for i in range(len(all_label)):
        label = all_label[i].numpy().tolist()
        label = int(label)
        if label in center_dict.keys():
            center_dict[label] = torch.cat((all_fea[i].reshape(1, -1), center_dict[label]), dim=0)
        else:
            center_dict[label] = all_fea[i].reshape(1, -1)

    for key, value in center_dict.items():
        center_dict[key] = torch.mean(value, dim=0)

    return center_dict