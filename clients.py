import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm
from dataset import MyDataset
from iou import calc_iou

class Client(object):
    def __init__(self, train_dataset, device, name, summary_writer=None):
        self.train_dataset = train_dataset
        self.device = device
        self.train_dataloader = None
        self.local_parameters = None
        self.name = name
        self.epoch_count = 0
        self.summary_writer = summary_writer

    def local_update(self, local_epoch, local_batchsize, net, loss_fun, optim, global_parameters):
        net.load_state_dict(global_parameters, strict=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=local_batchsize, shuffle=True)
        for epoch in range(local_epoch):

            metrics = {}
            metrics['loss'] = 0
            metrics['iou'] = 0
            epoch_samples = 0
            train_bar = tqdm.tqdm(self.train_dataloader)
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = net(inputs)
                loss = loss_fun(outputs, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()

                metrics['loss'] += loss.data.cpu().numpy() * inputs.size(0)
                metrics['iou'] += calc_iou(outputs.detach(), labels) * inputs.size(0)
                # statistics
                epoch_samples += inputs.size(0)
                train_bar.set_description("%s [%d|%d]: Loss: %.4f, IOU: %.4f" % (
                    self.name, epoch, local_epoch, metrics['loss'] / epoch_samples, metrics['iou'] / epoch_samples
                ))

            self.epoch_count += 1
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(tag="%s_train_loss" % self.name, scalar_value=metrics['loss'] / epoch_samples,
                                          global_step=epoch)
                self.summary_writer.add_scalar(tag="%s_train_iou" % self.name, scalar_value=metrics['loss'] / epoch_samples,
                                          global_step=epoch)

        return net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, num_of_clients, device, road_ids_map=None, summary_writer=None):
        self.num_of_clients = num_of_clients
        self.device = device
        self.clients_set = {}
        self.summary_writer = summary_writer
        self.test_dataloader = None

        # 每个客户端所分配到的数据文件夹编号
        self.data_ids_map = road_ids_map

        self.create_clients()
        self.load_test_data()

    def load_test_data(self):
        test_dataset = MyDataset(root_dir="./data/val", data_ids=list(range(3)))
        self.test_dataloader = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=2)

    def create_clients(self):
        for i in range(self.num_of_clients):

            # 为每个 client 创建数据集
            dataset = MyDataset(root_dir='./data', data_ids=self.data_ids_map[i])
            client = Client(dataset, self.device, name="client_%d" % i, summary_writer=self.summary_writer)
            self.clients_set[i] = client

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_ids_map = {
        0: [0, 1, 3],
        1: [4, 5, 6],
        2: [2, 7, 8]
    }
    MyClients = ClientsGroup(3, 1, data_ids_map)