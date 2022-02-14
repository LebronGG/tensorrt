import onnxruntime as ort
import time 
import cv2, copy
import numpy as np
import onnx
import sys

def rgb2yuv(rgb):
    m = np.array([[66, 129, 25],
                  [-38, -74, 112],
                  [112, -94, -18]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:, 0] += 16.
    ycbcr[:, 1:] += 128.
    np.clip(ycbcr, 0, 255)
    ycbcr = ycbcr.astype(np.uint8)
    return ycbcr.reshape(shape)


def yuv2rgb(yuv):
    m = np.array([[66, 129, 25],
                  [-38, -74, 112],
                  [112, -94, -18]])
    shape = yuv.shape
    if len(shape) == 3:
        yuv = yuv.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(yuv).astype(np.float32)
    rgb[:, 0] -= 16.
    rgb[:, 1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)

def preprocess(image):
    ycbcr_img_l = rgb2yuv(image).astype(np.uint8)
    input_y = ycbcr_img_l[..., 0]
    im_input = np.expand_dims(input_y, 2)
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]        
    im_input = np.asarray(im_input ,dtype=np.float16)
    return im_input

def postproess(out_image):
    out_image=out_image.squeeze(0)
    out_img = np.transpose(out_image, (1, 2, 0))
  #  output_folder = "onnx_output.png"
    #cv2.imwrite(output_folder, out_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    
def to_numpy(tensor):
   # return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    return tensor.cpu().numpy().astype(np.uint8)

pic_addr = "001441.png"
image = cv2.imread(pic_addr, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]
img = preprocess(image)

sess = ort.InferenceSession("../models/sr_torch1.onnx")
# sess.set_providers(['CPUExecutionProvider'])
# ort_session.set_providers(['CUDAExecutionProvider'], [{'device_id': 1}])

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

print(input_name, label_name)
# sys.exit()
for i in range(100):
    t1 = time.time()
    prob = sess.run([label_name], {input_name:img})[0]
    t2= time.time()
    print("---use_time---: ", t2-t1)

    





