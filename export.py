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

def drop():
    model = models.resnet152(pretrained=True)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 2)
    )

    # model.load_state_dict(torch.load('../models/drop.pth'))
    model.to(device)
    model.eval()

    input_name = ['input']
    output_name = ['output']

    data = torch.tensor(np.random.rand(1, 3, 224, 224), dtype=torch.float32).to(device)
    
    torch.save(model.state_dict(), '../models/resnet152.pth')

    torch.onnx.export(model, data, '../models/resnet152.onnx', 
                    export_params=True,        # store the trained parameter weights inside the model file
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    opset_version=11,          # the ONNX version to export the model to
                    verbose=False,
                    #   dynamic_axes={'input' : {0 : 'batch_size', 2: 'width', 3: 'height'}, 'output' : {0 : 'batch_size', 2: 'width', 3: 'height'}}
                    )

    test = onnx.load('../models/resnet152.onnx')
    onnx.checker.check_model(test)
    print("==> Passed")

drop()