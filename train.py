import torch.nn.functional as F
from torch import nn
import copy
import time
from collections import defaultdict
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import tqdm
from tensorboardX import SummaryWriter

from model import UNet
from dataset import MyDataset
from iou import calc_iou

def calc_loss(pred, target):
    bce = F.cross_entropy(pred, target)
    loss = bce

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    summary_writer = SummaryWriter(log_dir="runs")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # train
        scheduler.step()
        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])

        model.train()  # Set model to training mode

        metrics = defaultdict(float)
        metrics['loss'] = 0
        metrics['iou'] = 0
        epoch_samples = 0

        train_bar = tqdm.tqdm(train_dataloader)
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            loss = calc_loss(outputs, labels)

            metrics['loss'] += loss.data.cpu().numpy() * inputs.size(0)
            metrics['iou'] += calc_iou(outputs.detach(), labels) * inputs.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            epoch_samples += inputs.size(0)
            train_bar.set_description("Train [%d|%d]: Loss: %.4f, IOU: %.4f" % (
                epoch, num_epochs, metrics['loss'] / epoch_samples, metrics['iou'] / epoch_samples
            ))

        #print_metrics(metrics, epoch_samples, "train")
        summary_writer.add_scalar(tag="global_train_loss", scalar_value=metrics['loss'] / epoch_samples,
                                  global_step=epoch + 1)
        summary_writer.add_scalar(tag="global_train_iou", scalar_value=metrics['loss'] / epoch_samples,
                                  global_step=epoch + 1)
        if epoch % 20 == 0:
            torch.save(model.state_dict(), "ckpt/global/epoch_%d_loss_%.4f_iou_%.4f.pth" % (
                epoch, metrics['loss'] / epoch_samples, metrics['iou'] / epoch_samples
            ))

        # val
        model.eval()  # Set model to evaluate mode

        metrics = defaultdict(float)
        metrics['loss'] = 0
        metrics['iou'] = 0
        epoch_samples = 0

        val_bar = tqdm.tqdm(val_dataloader)
        for inputs, labels in val_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            with torch.no_grad():
                outputs = model(inputs)
                loss = calc_loss(outputs, labels)

                metrics['loss'] += loss.data.cpu().numpy() * inputs.size(0)
                metrics['iou'] += calc_iou(outputs.detach(), labels) * inputs.size(0)

            # statistics
            epoch_samples += inputs.size(0)
            val_bar.set_description("Val   [%d|%d]: Loss: %.4f, IOU: %.4f" % (
                epoch, num_epochs, metrics['loss'] / epoch_samples, metrics['iou'] / epoch_samples
            ))

        #print_metrics(metrics, epoch_samples, "val")
        epoch_loss = metrics['loss'] / epoch_samples
        epoch_iou = metrics['iou'] / epoch_samples
        summary_writer.add_scalar(tag="global_val_loss", scalar_value=metrics['loss'] / epoch_samples,
                                  global_step=epoch + 1)
        summary_writer.add_scalar(tag="global_val_iou", scalar_value=metrics['loss'] / epoch_samples,
                                  global_step=epoch + 1)

        # deep copy the model
        if epoch_loss < best_loss:
            print("saving best model")
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "ckpt/global/epoch_best.pth")


        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=1, n_classes=2, bilinear=False)
    model = model.to(device)

    train_dataset = MyDataset(root_dir="./data/local5", data_ids=list(range(2)))
    train_dataloader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=2)
    val_dataset = MyDataset(root_dir="./data/val", data_ids=list(range(1)))
    val_dataloader = DataLoader(val_dataset, batch_size=48, shuffle=False, num_workers=2)

    summary(model, input_size=(1, 256, 256))

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

    model = train_model(model, train_dataloader, val_dataloader, optimizer_ft, exp_lr_scheduler, num_epochs=100)