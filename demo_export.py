from torchvision import transforms, models
import torchvision
import torch
from torch.autograd import Variable
import onnx
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(torch.__version__)


input_name = ['input']
output_name = ['output']
input = Variable(torch.randn(1, 3, 224, 224)).cuda()
model = torchvision.models.mobilenet_v2(pretrained=True).cuda()
torch.onnx.export(model, input, './models/mobilenet_v2.onnx', input_names=input_name, output_names=output_name, verbose=True)

test = onnx.load('./models/mobilenet_v2.onnx')
onnx.checker.check_model(test)
print("==> Passed")

