#--*-- coding:utf-8 --*--
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pycuda.autoinit
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import tensorrt as trt
import time
import cv2
import base64

def init():   # 1. 子进程开始初始化cuda driver
    cuda.init()

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
TRT_LOGGER = trt.Logger()

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 

class TensorRTEngine(object):
    def __init__(self, onnx_file, engine_file, batch_size=1):
        #2. trt engine创建前首先初始化cuda上下文
        self.cfx = cuda.Device(0).make_context()  
        self.engine = self.load_engine(onnx_file, engine_file, batch_size)
        # self.input_shape, self.output_shape = self.infer_shape()
          
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()
        
        # PyCUDA ERROR: The context stack was not empty upon module cleanup.
        self.cfx.pop() 
        
        self.shape_of_output = (batch_size, 2)

    def __del__(self):
        del self.inputs
        del self.outputs
        del self.stream
        # 2. 实例释放时需要detech cuda上下文
        self.cfx.detach() 

    def load_engine(self, onnx_file, engine_file, batch_size=1):

        def build_engine(batch_size, save_engine):
            EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
                builder.max_batch_size = batch_size
                builder.max_workspace_size = 1 << 30
                builder.fp16_mode = True
                with open(onnx_file, 'rb') as model:
                    if not parser.parse(model.read()):
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                engine = builder.build_cuda_engine(network)
            print("Load onnx sucessful!")
            if save_engine:
                with open(engine_file, "wb") as f:
                    f.write(engine.serialize())
            return engine
        
        if os.path.exists(engine_file):
            print("Reading engine from file {}".format(engine_file))
            with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine(batch_size, True)

    def infer_shape(self):
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                input_shape = self.engine.get_binding_shape(binding)
            else:
                output_shape = self.engine.get_binding_shape(binding)
        
        return input_shape, output_shape
    
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def do_inference(self, context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def preprocess(self, image):
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
        return img_np_nchw.reshape(-1)

    def postprocess(self, data):
        h_outputs = data.reshape(self.shape_of_output)
        return h_outputs

    def inference_file(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.inputs[0].host = self.preprocess(image)
        # self.cfx.push()  # 3. 推理前执行cfx.push()
        trt_outputs = self.do_inference(self.context, bindings=self.bindings, 
                                            inputs=self.inputs, 
                                            outputs=self.outputs, 
                                            stream=self.stream)
        
        output = self.postprocess(trt_outputs[0])
        # self.cfx.pop()  # 3. 推理后执行cfx.pop()
        return softmax(output)
    
    def inference_image(self, image):
        self.inputs[0].host = self.preprocess(image)
        self.cfx.push()  # 3. 推理前执行cfx.push()
        trt_outputs = self.do_inference(self.context, bindings=self.bindings, 
                                            inputs=self.inputs, 
                                            outputs=self.outputs, 
                                            stream=self.stream)
        
        output = self.postprocess(trt_outputs[0])
        self.cfx.pop()  # 3. 推理后执行cfx.pop()
        return softmax(output)

if __name__ == '__main__':
    model = TensorRTEngine('../models/drop.onnx', '../models/drop_fp16.trt')
    image = cv2.imread('1.png')
    
    
    if True:
        image = cv2.imencode('.png', image)[1]
        imgencode = base64.b64encode(image).decode('utf8')

        imgdecode = base64.b64decode(imgencode.encode('utf8'))
        image = np.fromstring(imgdecode, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(10):
        t1 = time.time()
        res = model.inference_image(image)
        t2 = time.time()
        print(t2 - t1, res[0])