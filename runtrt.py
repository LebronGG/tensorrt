#--*-- coding:utf-8 --*--
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import time
from PIL import Image
import cv2
from torchvision import transforms


TRT_LOGGER = trt.Logger()  # This logger is required to build an engine

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", \
               fp16_mode=False, int8_mode=False, save_engine=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            # Your workspace size 1 << 30 # 256MiB
            builder.max_workspace_size = 1 << 30
            builder.max_batch_size = max_batch_size
            # pdb.set_trace()
            builder.fp16_mode = fp16_mode  # Default: False
            builder.int8_mode = int8_mode  # Default: False
            if int8_mode:
                #To be updated
                raise NotImplementedError

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    t1 = time.time()

    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    t2 = time.time()

    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    t3 = time.time()

    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    t4 = time.time()

    # Synchronize the stream
    stream.synchronize()
    t5 = time.time()

    print("Transfer:{:.6f} inference:{:6f} back:{:6f} synchronize:{:6f}".format(t2 - t1, t3 - t2, t4 - t3, t5 - t3))

    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

def torch_trans(image):
    tfms = transforms.Compose([ transforms.ToTensor(), transforms.Resize((224,224)),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_np_nchw = tfms(image).unsqueeze(0).cpu().numpy().astype(dtype=np.float32)
    return img_np_nchw

def np_trans(image):
    image_cv = cv2.resize(image, (224, 224))
    miu = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = np.array(image_cv, dtype=np.float)/255.
    img_np = img_np.transpose((2, 0, 1))
    img_np -= miu
    img_np /= std
    img_np_nchw = img_np[np.newaxis]
    img_np_nchw = np.tile(img_np_nchw,(1, 1, 1, 1))
    img_np_nchw = img_np_nchw.astype(dtype=np.float32)
    return img_np_nchw

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 


max_batch_size = 1
shape_of_output = (max_batch_size, 2)

onnx_model_path = "../models/drop.onnx"
trt_engine_path = "../models/drop_fp16.trt"

engine = get_engine(max_batch_size, onnx_model_path, trt_engine_path, fp16_mode=True, int8_mode=False)
# Create the context for this engine
context = engine.create_execution_context()
# Allocate buffers for input and output
inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

image = cv2.cvtColor(cv2.imread('1.png'), cv2.COLOR_BGR2RGB)

inputs[0].host = np_trans(image).reshape(-1)

# inputs[1].host = ... for multiple input
for i in range(10):
    t1 = time.time()
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    prob = postprocess_the_outputs(trt_outputs[0], shape_of_output)
    print(round(time.time() - t1, 6), softmax(prob))


