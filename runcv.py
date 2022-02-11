import time
import cv2 as cv
import numpy as np
import os
 

onnx_name = "../models/drop.onnx"
 
def BlobFromNormalizeRGBImage(img, meanList, stdList):
    img = cv.resize(img, (224, 224))
    img = img / 255.0        # 归一化至0 ~ 1区间
    R, G, B = cv.split(img)
    if meanList is not None:
        R = R - meanList[0]
        G=  G - meanList[1]
        B = B - meanList[2]
 
    if stdList is not None:
        R = R / stdList[0]
        G = G / stdList[1]
        B = B / stdList[2]
    
    # 通道合并
    merged = cv.merge([R, G, B])
    merged = merged.transpose((2, 0, 1))
    blob = np.expand_dims(merged, 0)   
    return blob
 
def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 

def opencv_onnx_defect_infer():
    net = cv.dnn.readNetFromONNX(onnx_name)

    image = cv.cvtColor(cv.imread('2.png'), cv.COLOR_BGR2RGB)
    
    blob = BlobFromNormalizeRGBImage(image, meanList=[0.485, 0.456, 0.406],stdList=[0.229, 0.224, 0.225])
    # print("BlobFromNormalizeRGBImage:",blob.shape) # (1, 3, 300, 300)

    # 设置模型的输入
    net.setInput(blob)
    for i in range(10):
        t1 = time.time()
        out = net.forward().flatten()
        print(time.time() - t1, softmax(out))

opencv_onnx_defect_infer()