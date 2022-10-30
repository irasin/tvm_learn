import os
import glob

import cv2
import numpy as np
from torch.utils.data import Dataset

from ..metric import letterbox, xywhn2xyxy, xyxy2xywhn
from ...cv_dataset import IMG_FORMATS


class DetectionCOCO128Dataset(Dataset):

    def __init__(
        self,
        image_dir_path,
        label_dir_path,
        image_size=640,
        stride=32,
    ):

        self.image_dir_path = image_dir_path
        self.label_dir_path = label_dir_path
        self.image_size = image_size
        self.stride = stride

        self.img_files, self.label_files = [], []

        ## assert all image files are under the image_dir_path with valid suffix
        all_img_files = glob.glob(os.path.join(self.image_dir_path, "*.*"))
        all_img_files = sorted(
            x for x in all_img_files
            if os.path.splitext(x)[-1][1:].lower() in IMG_FORMATS)

        ## assert all image files has corresponding label file with the same basename but .txt suffix
        all_label_files = glob.glob(os.path.join(self.label_dir_path, "*.*"))

        for img_path in all_img_files:
            label_file = os.path.join(
                self.label_dir_path,
                os.path.splitext(os.path.basename(img_path))[0] + ".txt")
            if label_file in all_label_files:
                self.img_files.append(img_path)
                self.label_files.append(label_file)

        assert len(self.img_files) == len(self.label_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # load image
        img, (h0, w0), (h, w) = self._load_image(index)

        # letterbox
        img, ratio, pad = letterbox(img, self.image_size)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # load label
        raw_label = self._load_txt_label(index)
        # import pdb;pdb.set_trace()
        # normalized xywh to pixel xyxy format
        raw_label[:, 1:] = xywhn2xyxy(raw_label[:, 1:],
                                      ratio[0] * w,
                                      ratio[1] * h,
                                      padw=pad[0],
                                      padh=pad[1])

        raw_label[:, 1:] = xyxy2xywhn(raw_label[:, 1:],
                                      w=img.shape[1],
                                      h=img.shape[0],
                                      clip=True,
                                      eps=1E-3)

        nl = len(raw_label)  # number of labels
        labels_out = np.zeros((nl, 6))
        labels_out[:, 1:] = raw_label

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.0  # 0~1 np array

        return img, labels_out, self.img_files[index], shapes

    def _load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)

        im = cv2.imread(self.img_files[i])  # BGR
        h0, w0 = im.shape[:2]  # orig hw
        r = self.image_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_LINEAR)
        return im.astype("float32"), (
            h0, w0), im.shape[:2]  # im, hw_original, hw_resized

    def _load_txt_label(self, i):
        # import pdb;pdb.set_trace()
        with open(self.label_files[i]) as f:
            data = f.read().splitlines()

        labels = []
        for i in data:
            labels.append([float(j) for j in i.split()])

        return np.array(labels)

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        # return torch.stack(im, 0).numpy(), torch.cat(label, 0).numpy(), path, shapes
        # import pdb;pdb.set_trace()
        return np.concatenate([i[None] for i in im],
                              axis=0), np.concatenate(label, 0), path, shapes
