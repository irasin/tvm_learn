#%%
import tvm
import tvm.relay as relay

# %%
model_path = "/home/cc/code/yolo/yolov5/yolov5s.onnx"

import onnx

model = onnx.load(model_path)

input_name = "images"
input_shape = (1, 3, 640, 640)

shape_dict = {input_name: input_shape}

mod, params = relay.frontend.from_onnx(model, shape_dict)

# %%
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# %%
lib.export_library("/home/cc/code/yolo/yolov5/yolov5s.so")
# %%
