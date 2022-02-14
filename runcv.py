import time
import cv2
import numpy as np
import os
 
def BlobFromNormalizeRGBImage(img, meanList, stdList):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0        # 归一化至0 ~ 1区间
    R, G, B = cv2.split(img)
    if meanList is not None:
        R = R - meanList[0]
        G=  G - meanList[1]
        B = B - meanList[2]
 
    if stdList is not None:
        R = R / stdList[0]
        G = G / stdList[1]
        B = B / stdList[2]
    
    # 通道合并
    merged = cv2.merge([R, G, B])
    merged = merged.transpose((2, 0, 1))
    blob = np.expand_dims(merged, 0)   
    return blob
 
def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 

def opencv2_onnx_defect_infer():
    onnx_name = "../models/drop.onnx"
    net = cv2.dnn.readNetFromONNX(onnx_name)

    image = cv2.cvtColor(cv2.imread('2.png'), cv2.COLOR_BGR2RGB)
    blob = BlobFromNormalizeRGBImage(image, meanList=[0.485, 0.456, 0.406],stdList=[0.229, 0.224, 0.225])
    
    net.setInput(blob)
    for i in range(10):
        t1 = time.time()
        out = net.forward().flatten()
        print(time.time() - t1, softmax(out))

opencv2_onnx_defect_infer()