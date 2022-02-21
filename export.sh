# /opt/tensorrt/bin/trtexec --onnx=super_resolution.onnx --saveEngine=srn_fp16.trt --fp16  --workspace=1024

/opt/TensorRT-7.2.3.4/bin/trtexec --onnx=../models/drop.onnx --saveEngine=../models/drop.trt

# /opt/tensorrt/bin/trtexec --loadEngine=srn.trt --batch=256