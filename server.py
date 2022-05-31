import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
from model import UNet
from clients import ClientsGroup, Client
from iou import calc_iou
from torch.autograd import Variable

def calc_loss(pred, target):
    bce = F.cross_entropy(pred, target)
    loss = bce
    return loss

def calc_loss_focal(pred, target):
    focal = FocalLoss(gamma = 2.0, alpha = 1.0)
    loss = focal(pred, target)
    return loss
    
class FocalLoss1(nn.Module):
    #class > 2
    def __init__(self, weight=None, reduction='mean', gamma=1, eps=1e-6):
        super(FocalLoss1, self).__init__()
        #default setting is crosentropyloss
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return 10 * loss.mean()
    
class FocalLoss(nn.Module):
    #class = 2
    def __init__(self, weight=None, reduction='mean', gamma=1, alpha= 1.0, eps=1e-6):
        super(FocalLoss, self).__init__()
        #default setting is crosentropyloss
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp * self.alpha
        return 10 * loss.mean()
        
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-num_of_clients', type=int, default=9, help='numer of the clients')
parser.add_argument('-cfraction', type=float, default=1.0, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-batchsize', type=int, default=16, help='local train batch size')
parser.add_argument('-learning_rate', type=float, default=0.0001, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-val_freq', type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-save_freq', type=int, default=1, help='global model save frequency(of communication)')
parser.add_argument('-num_comm', type=int, default=12, help='number of communications')
parser.add_argument('-save_path', type=str, default='./ckpt/ablation/clientnum=9', help='the saving path of checkpoints')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__
    client_weight = [0.32495, 0.39730, 0.27775]
    test_mkdir(args['save_path'])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    summary_writer = SummaryWriter(log_dir="runs_fedavg")

    # 创建服务器模型
    net = UNet(n_channels=1, n_classes=2, bilinear=False)
    net = net.to(device)

    # 定义损失函数
    loss_func = calc_loss_focal
    optim = optim.Adam(net.parameters(), lr=args['learning_rate'])

    # 创建客户端组
    data_ids_map = {
        0: [0, 1, 3],
        1: [4, 5, 6],
        2: [2, 7, 8]
    }
    
    data_ids_map1 = {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
        3: [6, 7],
        4: [8]
    }
    
    data_ids_map2 = {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
        4: [4],
        5: [5],
        6: [6],
        7: [7],
        8: [8]
    }
    client_group = ClientsGroup(args['num_of_clients'], device, data_ids_map2, summary_writer)
    test_dataloader = client_group.test_dataloader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    # 复制一份全局网络参数
    global_parameters = {}
    best_parameters = {}
    best_loss = 2.0
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    # 进行通信更新
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))
        # 随机选择一部分Client，全部选择会增大通信量，且实验效果可能会不好
        clients_in_comm = range(num_in_comm)
        sum_parameters = None
        # 每个Client基于当前模型参数和自己的数据训练并更新模型，返回每个Client更新后的参数
        for client in clients_in_comm:
            local_parameters = client_group.clients_set[client].local_update(args['epoch'], args['batchsize'], net,
                                                                         loss_func, optim, global_parameters)
            # 对每个 Client 返回的参数进行相加
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        # 求参数平均值，并更新全局参数（服务器参数）
        for var in global_parameters:
            global_parameters[var] = sum_parameters[var] / num_in_comm * 1.0

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                metrics = {}
                metrics['loss'] = 0
                metrics['iou'] = 0
                epoch_samples = 0
                test_bar = tqdm(test_dataloader)
                for inputs, labels in test_bar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    loss = loss_func(outputs, labels)
                    metrics['loss'] += loss.data.cpu().numpy() * inputs.size(0)
                    metrics['iou'] += calc_iou(outputs.detach(), labels) * inputs.size(0)
                    # statistics
                    epoch_samples += inputs.size(0)
                    test_bar.set_description("%s [%d|%d]: Loss: %.4f, IOU: %.4f" % (
                        "Server", i, args['num_comm'], metrics['loss'] / epoch_samples, metrics['iou'] / epoch_samples
                    ))
                summary_writer.add_scalar(tag="%global_val_loss",
                                                scalar_value=metrics['loss'] / epoch_samples,
                                                global_step=i + 1)
                summary_writer.add_scalar(tag="%global_val_iou",
                                                scalar_value=metrics['iou'] / epoch_samples,
                                                global_step=i + 1)
                #updating best model parameters
                if best_loss > metrics['loss'] / epoch_samples:
                    best_loss = metrics['loss'] / epoch_samples
                    print('saving best')
                    for key, var in global_parameters.items():
                        best_parameters[key] = var.clone()
                if best_loss < metrics['loss'] / epoch_samples - 0.02:
                    print(best_loss, metrics['loss'] / epoch_samples)
                    print('tracing back model')
                    for key, var in best_parameters.items():
                        global_parameters[key] = var.clone()
                        
            if (i + 1) % args['save_freq'] == 0:
                torch.save(net, os.path.join(args['save_path'],
                                             'Server_comm{}_E{}_LOSS{}_B{}_lr{}_num_clients{}_cf{}'.format(
                                                    i, args['epoch'],
                                                    metrics['loss'] / epoch_samples,
                                                    args['batchsize'],
                                                    args['learning_rate'],
                                                    args['num_of_clients'],
                                                    args['cfraction'])
                                             ))