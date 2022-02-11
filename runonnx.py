from time import time
import onnxruntime as rt
import numpy as  np
import time
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def torch_trans(image):
    tfms = transforms.Compose([ transforms.ToTensor(), transforms.Resize((224,224)),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_np_nchw = tfms(image).unsqueeze(0).cpu().numpy().astype(dtype=np.float32)
    return img_np_nchw

def np_trans(image):
    image_cv = cv2.resize(image, (224, 224))
    miu = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = np.array(image_cv, dtype=np.float)/255.
    img_np = img_np.transpose((2, 0, 1))
    img_np -= miu
    img_np /= std
    img_np_nchw = img_np[np.newaxis]
    img_np_nchw = np.tile(img_np_nchw,(1, 1, 1, 1))
    img_np_nchw = img_np_nchw.astype(dtype=np.float32)
    return img_np_nchw

def cv_trans(image):
    meanList=[0.485, 0.456, 0.406]
    stdList=[0.229, 0.224, 0.225]
    img = cv2.resize(image, (224, 224))
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
    blob = np.expand_dims(merged, 0).astype(dtype=np.float32)
    return blob


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 

sess = rt.InferenceSession('../models/resnet152.onnx')

# cpu gpu setting
sess.set_providers(['CPUExecutionProvider'])

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


# image = Image.open('2.png').convert('RGB')
image = cv2.cvtColor(cv2.imread('2.png'), cv2.COLOR_BGR2RGB)

img = np_trans(image)

for i in range(1000):
    t1 = time.time()
    prob = sess.run([label_name], {input_name:img})[0]
    print(time.time() - t1, softmax(prob))