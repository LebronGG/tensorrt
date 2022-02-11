import torch
from torchvision import models
import time, os
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class resnet():
    def __init__(self):
        self.device = torch.device('cuda')

        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 2)
        )

        self.model.load_state_dict(torch.load('../models/drop.pth'))

        self.model.eval()
        self.model.to(self.device)

        self.tfms = transforms.Compose([ transforms.ToTensor(), transforms.Resize((224,224)),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(self, image):
        img = self.tfms(image).unsqueeze(0).to(self.device)
        logits = self.model(img)
        prob = logits.data.cpu().numpy()
        return prob


# image = Image.open('2.png').convert('RGB')
image = cv2.cvtColor(cv2.imread('1.png'), cv2.COLOR_BGR2RGB)

model = resnet()

with torch.no_grad():
    for i in range(10):
        t1 = time.time()
        prob = model.predict(image)
        print(time.time() - t1, prob)