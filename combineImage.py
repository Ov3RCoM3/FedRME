import os
import cv2
import numpy as np
from PIL import Image

IMG_WIDTH = 7056
IMG_HEIGHT = 3439
ROOT = "F:/test数据生成(road2、11)/testdata/0/测试结果"

def combine(comImg, filename):
    h_num = int(float(filename.split("_")[0]))
    temp  = filename.split("_")[1]
    w_num = int(float(temp.split("_")[0]))
    path = os.path.join(ROOT, filename)
    im = Image.open(path)
    im = im.getdata()
    im = np.reshape(im, (256, 256))
    for w in range(256):
        for h in range(256):
            comImg[h_num + h][w_num + w] = im[h][w]
    

def main():
    combineImg = np.zeros(shape = (IMG_HEIGHT, IMG_WIDTH))
    fileList = os.listdir(ROOT)
    for filename in fileList:
        print(filename)
        combine(combineImg, filename)
        
    res = Image.fromarray(combineImg.astype(np.int8), mode='L')
    res.save("test-result-road6.png")
        
if __name__ == "__main__":
    main()