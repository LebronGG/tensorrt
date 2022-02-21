from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import os, base64
from io import BytesIO
import cv2
import numpy as np
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


app = Flask(__name__)


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 


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

        self.tfms = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(self, image):
        img = self.tfms(image).unsqueeze(0).to(self.device)
        logits = self.model(img)
        prob = logits.data.cpu().numpy()
        return softmax(prob)

@app.route('/', methods=['GET'])
def index():
    return 'ok'

model = resnet()

@app.route('/predict', methods=['GET', 'POST'])
def getImg():
    t1 = time.time()
    img_b64decode = request.form["data"]
    imgData = base64.b64decode(img_b64decode.encode('utf8'))
    img_array = np.fromstring(imgData, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    t2 = time.time()

    output = model.predict(image)
    data = {'conf': output.tolist()}
    return data

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
