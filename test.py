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
onnx = 'http://127.0.0.1:5051/getImg'
cuda = 'http://127.0.0.1:5052/getImg'
# gpu = 'http://thservinggpu.cc.163.com/predictions/porndet'
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
    p = concurrent.futures.ThreadPoolExecutor(max_workers=4) # ProcessPoolExecutor ThreadPoolExecutor
    image = cv2.imread(file)
    # image = cv2.resize(image, (224, 224))
    image = cv2.imencode('.png', image)[1]
    img_b64encode = base64.b64encode(image).decode('utf8')

    data = {"image": img_b64encode}
    jdata = {"data": img_b64encode}


    t1 = time.time()

    # files = [data for i in range(100)]
    # res = list(p.map(test_onnx, files))

    for i in range(10):
        t2 = time.time()
        res = test_cpu(jdata)
        print(time.time() - t2, res)

    # files = [jdata for i in range(100)]
    # res = list(p.map(test_cpu, files))

    
    # print(res)
    print(time.time() - t1)