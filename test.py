import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import time
import json
import concurrent.futures

sess = requests.Session()

file = '1.png'
onnx = 'http://127.0.0.1:222/predict'
cuda = 'http://127.0.0.1:5001/predict'
gpu = 'http://10.170.100.12:19180/predictions/porndet'
cpu = 'http://thserving.cc.163.com:18080/predictions/porndet'
# cpu = 'http://thserving-ol.cc.163.com:18080/predictions/pornmodel'

def test_onnx(data):
    res = sess.post(onnx, data=data, timeout=2).json()
    return res

def test_cuda(data):
    res = sess.post(cuda, data=data, timeout=2).json()
    return res

def test_gpu(data):
    res = sess.post(gpu, json=data, timeout=2).json()
    return res

def test_cpu(data):
    res = sess.post(cpu, json=data, timeout=2).json()
    return res


if __name__ == '__main__':
    p = concurrent.futures.ProcessPoolExecutor(max_workers=30) # ProcessPoolExecutor ThreadPoolExecutor
    image = cv2.imread(file)
    image = cv2.resize(image, (224, 224))
    image = cv2.imencode('.png', image)[1]
    img_b64encode = base64.b64encode(image).decode('utf8')

    data = {"data": img_b64encode}

    files = [data for i in range(2000)]
    
    t1 = time.time()
    
    res = list(p.map(test_onnx, files))
    
    # for file in files:
    #     t2 = time.time()
    #     res = test_onnx(file)
    #     print(time.time() - t2, res)

    print(time.time() - t1)
    