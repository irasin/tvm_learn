# We can create a dataloader with two aspects, the dataset itself and the specific task,
# e.g. COCO + detection, COCO + segmentation, ImageNet + classification.

# Given the dataset and the task, the DL models can be very different,
# but the processing to generate input, postprocess output etc. should be the same.
# The key idea here is that we could create a workload for preprocessing and postprocessing.
# More specifically, given dataset and task and some other extra info(e.g. batchsize, threadhold, etc.),
# we generate a dataloader and evaluator.

# Fortunately, pytorch provides very user-friendly dataloader interface for us,
# we only need to implement the part of dataset processing with given dataset and task.
# For compatibility, we return numpy array instead of torch tensor.

# The evaluator provides functionality for visualization, computational accuracy and performance evaluation.

# Examples are as follows
# assert the dataset itself is under the fast_deploy/dataset directory

# dataset = "coco128"
# task = "detection"
# extra_info = { ## extra_info contains all information to create dataloader and evaluator
#     "image_path": "xxx",
#     "label_path": "yyy",
#     "batchsize": 32,
#     "shuffle": False
# }

# loader, evaluator = create_data_workload(dataset, task, extra_info)

# for batch_idx, all_input in enumerate(loader):
#     images = all_input[0]
#     targets = all_input[1]
#     # ...
#     # do inference
#     model = ...
#     t1 = time.time()
#     preds = model(images, ...)
#     t2 = time.time()
#     inferr_time = t2 - t1

#     # for object detection, the evaluate method will plot the result and calculate accuracy, etc.

#     evaluate_data = [preds, targets, inferr_time, ...]
#     evaluator.evaluate(evaluate_data)

## show final evaluation info, for object detection, summary will show mAP and performance info, etc.
# evaluator.summary()

###################################################################################################

from .cv import create_coco128_detection_workload

SUPPORTED_DATA_WORKLOAD = {
    "coco128_detection": create_coco128_detection_workload,
    "imagenet_classification": None,
}


def create_data_workload(dataset, task, extra_info_dict):
    print(extra_info_dict)

    dataset = dataset.lower()
    task = task.lower()
    pattern = f"{dataset}_{task}"

    creator = SUPPORTED_DATA_WORKLOAD.get(pattern, None)
    assert creator, f"dataset: {dataset} with task: {task} has not been supported"

    return creator(extra_info_dict)
