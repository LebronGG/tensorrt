from statistics import mode
import torchvision
import torch
import onnx
import os
from torchvision import transforms, models
from PIL import Image
import numpy as np
import torch.nn as nn


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
model.to(device)
model.eval()

input_name = ['input']
output_name = ['output']

data = torch.rand(1, 3, 224, 224).to(device)

torch.onnx.export(model, data, '../models/drop.onnx', input_names=input_name, output_names=output_name, verbose=False)

test = onnx.load('../models/drop.onnx')
onnx.checker.check_model(test)
print("==> Passed")

