# -*- coding: utf-8 -*-

from PIL import Image
from pylab import *

def compare_pic_L(pic1,pic2):
    TP=0
    FP=0
    FN=0
    im1 = Image.open(pic1).convert('L')
    w,h =im1.size
    print('size=',w*h)
    w=int(w)
    h=int(h)
    aim1 = np.transpose(array(im1))
    im2 = Image.open(pic2).convert('L')
    aim2 = np.transpose(array(im2))
    for i in range(w):
        for j in range(h):
            if aim1[i][j] == 255 and aim2[i][j] == 255:
                TP = TP+1
            elif aim1[i][j] == 255 and aim2[i][j] == 0:
                FP = FP+1
            elif aim1[i][j] == 0 and aim2[i][j] == 255:
                FN = FN+1
    print(TP,FP,FN)
    prec=float(TP/(TP+FP))
    recall=float(TP/(TP+FN))
    F1=float(2*prec*recall/(prec+recall))
    print('precision=',prec*100)
    print('recall=',recall*100)
    print('F1-score=',F1)
    


if __name__ == "__main__":
    compare_pic_L('groundtruth.png','test-result.png')