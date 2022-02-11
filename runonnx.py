from time import time
import onnxruntime as rt
import numpy as  np
import time
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

sess = rt.InferenceSession('../models/drop.onnx')

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


tfms = transforms.Compose([ transforms.ToTensor(), transforms.Resize((224,224)),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# image = Image.open('2.png').convert('RGB')
image = cv2.cvtColor(cv2.imread('1.png'), cv2.COLOR_BGR2RGB)

for i in range(10):
    t1 = time.time()
    pred_onx = sess.run([label_name], {input_name:tfms(image).unsqueeze(0).cpu().numpy()})[0]
    t2 = time.time()
    print(t2 - t1, pred_onx)