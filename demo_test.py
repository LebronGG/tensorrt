import torch
from torchvision import models
import time, os
from torch.autograd import Variable
# import tensorrt
# print(tensorrt.__version__)
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

net = models.resnet50().cuda()
x = Variable(torch.randn(1, 3, 224, 224)).cuda()

with torch.no_grad():
    for i in range(10):
        t1 = time.time()
        y_pt = net(x)
        print(time.time() - t1)