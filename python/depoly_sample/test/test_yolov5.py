import sys

sys.path.append("..")

import tvm

from fast_deploy.convert.onnx_to_igie import onnx2igie
from fast_deploy.data_workload.create_data_workload import create_data_workload
from fast_deploy.misc.timer import Timer

onnx_model_path = "/home/cc/code/yolo/yolov5/yolov5m.onnx"
input_name = "images"
batch_size = 32
image_size = 640
dtype = "float32"
deploy_target = "llvm"

dataset = "coco128"
task = "detection"
workload_info = {
    "image_dir_path":
    "/home/cc/code/tvm_learn/python/depoly_sample/datasets/coco128/images/train2017",
    "label_dir_path":
    "/home/cc/code/tvm_learn/python/depoly_sample/datasets/coco128/labels/train2017",
    "batch_size": batch_size,
}

# get module/device
shape_dict = {input_name: (batch_size, 3, image_size, image_size)}
m, device = onnx2igie(onnx_model_path, shape_dict, batch_size, deploy_target)

# get dataloader/evaluator
dataloader, evaluator = create_data_workload(dataset, task, workload_info)

timer = Timer()
for batch_id, all_inputs in enumerate(dataloader):

    ## set input
    im = all_inputs[
        0]  ## the creator of dataloader should know what all_inputs contains
    m.set_input(input_name, tvm.nd.array(im, device))
    device.sync()

    print(f"batchid = {batch_id}, im.shape = {im.shape}")

    ## run
    # with timer.timeit_sync(device):
    with timer.timeit():
        m.run()

    ## get output
    num_outputs = m.get_num_outputs()
    all_outputs = []
    for i in range(num_outputs):
        all_outputs.append(m.get_output(i).asnumpy())

    ## evaluate batch result
    evaluate_data = [all_outputs, all_inputs, timer.last_duration]
    evaluator.evaluate(evaluate_data)

evaluator.summary()
