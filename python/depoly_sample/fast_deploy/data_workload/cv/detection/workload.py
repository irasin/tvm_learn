from .coco128.dataset import DetectionCOCO128Dataset
from .coco128.evaluator import DetectionCOCO128Evaluator

from ..cv_dataloader import NormalCVDataLoader


def create_coco128_detection_workload(workload_info):

    # create dataloader
    image_dir_path = workload_info.get("image_dir_path")
    label_dir_path = workload_info.get("label_dir_path")
    image_size = workload_info.get("image_size", 640)
    stride = workload_info.get("stride", 32)
    batch_size = workload_info.get("batch_size")
    shuffle = workload_info.get("shuffle", False)

    dataset = DetectionCOCO128Dataset(image_dir_path=image_dir_path,
                                      label_dir_path=label_dir_path,
                                      image_size=image_size,
                                      stride=stride)

    batch_size = min(batch_size, len(dataset))
    drop_last = True if len(dataset) % batch_size else False

    loader = NormalCVDataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=drop_last,
                                collate_fn=dataset.collate_fn)

    conf_thres = workload_info.get("conf_thres", 0.001)
    iou_thres = workload_info.get("iou_thres", 0.065)

    evaluator = DetectionCOCO128Evaluator(conf_thres=conf_thres,
                                          iou_thres=iou_thres)

    return loader, evaluator
