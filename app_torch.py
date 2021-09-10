from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms, models
import os, base64
from io import BytesIO

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


app = Flask(__name__)

class resnet():
    def __init__(self):

        self.model = models.resnet50(pretrained=False)
        self.model.eval()
        self.model.cuda()

        self.tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(self, image):
        img = self.tfms(image).unsqueeze(0).cuda()
        logits = self.model(img)
        index = logits.data.cpu().numpy()
        return index

@app.route('/', methods=['GET'])
def index():
    return 'ok'

model = resnet()

@app.route('/getImg', methods=['GET', 'POST'])
def getImg():
    img_b64decode = request.form["image"]
    imgData = base64.b64decode(img_b64decode.encode('utf8'))

    image = Image.open(BytesIO(imgData)).convert('RGB')

    print(image.size)
    output = model.predict(image)
    data = {'conf': output.shape}
    return data

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5052, debug=False, processes=True,threaded=True)

