import torch
from torchvision import models
import time, os
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 


device = torch.device('cuda')

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 2)
)

model.load_state_dict(torch.load('../models/drop.pth'))

model.eval()
model.to(device)

tfms = transforms.Compose([  transforms.ToTensor(), transforms.Resize((224,224)),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

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


image = Image.open('1.png').convert('RGB')
# image = cv2.cvtColor(cv2.imread('1.png'), cv2.COLOR_BGR2RGB)
img = tfms(image).unsqueeze(0)
# img = torch.from_numpy(np_trans(image))


with torch.no_grad():
    for i in range(10):
        t1 = time.time()
        img = img.to(device)
        t2 = time.time()
        prob = model(img)
        t3 = time.time()
        res = prob.cpu().numpy()
        t4 = time.time()
        print("Transfer:{:.5f} inference:{:5f} back:{:5f}".format(t2 - t1, t3 - t2, t4 - t3))
        print(time.time() - t1, softmax(res))