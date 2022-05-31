from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

class MyDataset(Dataset):

    def __init__(self, root_dir, data_ids):

        self.root_dir = root_dir

        self.image_list = []
        self.label_list = []
        for i in data_ids:
            image_dir = os.path.join(self.root_dir, str(i), "surface")
            label_dir = os.path.join(self.root_dir, str(i), "marking")
            for image_name in os.listdir(image_dir):
                image_path = os.path.join(image_dir, image_name)
                label_path = os.path.join(label_dir, image_name)
                self.image_list.append(image_path)
                self.label_list.append(label_path)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        label = Image.open(self.label_list[idx]).convert("L")

        image = self.transform(image)
        label = np.array(label)
        label[label > 0] = 1
        label = torch.tensor(label).long()

        return image, label

if __name__ == "__main__":
    dataset = MyDataset('./data', [0, 1])
    dataset[0]