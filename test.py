import torch
from PIL import Image
import os
from torchvision import transforms
import cv2

from model import UNet

model_path = "./ckpt/fed/server_best.pth"

test_image_dir = "./testdata/surface1"
result_image_dir = "./result/surface1"
# test_image_dir = "./data/0/surface"
# result_image_dir = "./result/0/surface"

os.makedirs(result_image_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型
model = UNet(n_channels=1, n_classes=2, bilinear=False)
model = torch.load(model_path, map_location="cpu")
model = model.to(device)

# 获取测试图像路径集合
test_image_name_list = os.listdir(test_image_dir)
test_image_path_list = [os.path.join(test_image_dir, image_name) for image_name in test_image_name_list]
result_image_path_list = [os.path.join(result_image_dir, image_name) for image_name in test_image_name_list]

transform = transforms.Compose([
    transforms.ToTensor()
])
for i, image_path in enumerate(test_image_path_list):
    with torch.no_grad():
        image = Image.open(image_path)
        image = transform(image)

        image = image.unsqueeze(dim=0).to(device)

        output = model(image)
        predict = torch.softmax(output, dim=1)
        _, predict = torch.topk(predict, 1, dim=1)

        result = predict[0][0].cpu().numpy() * 255
        result = result.astype("uint8")

        cv2.imwrite(result_image_path_list[i], result)
