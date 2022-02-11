import torch
from torchvision import models
import time, os
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 


device = torch.device('cpu')

model = models.resnet50(pretrained=False)
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

tfms = transforms.Compose([ transforms.ToTensor(), transforms.Resize((224,224)),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# image = Image.open('2.png').convert('RGB')
image = cv2.cvtColor(cv2.imread('1.png'), cv2.COLOR_BGR2RGB)
img = tfms(image).unsqueeze(0).to(device)


with torch.no_grad():
    for i in range(10):
        t1 = time.time()
        prob = model(img).cpu().numpy()
        print(time.time() - t1, softmax(prob))