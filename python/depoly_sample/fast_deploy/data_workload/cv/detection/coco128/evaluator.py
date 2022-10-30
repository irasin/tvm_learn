import os
import cv2
import numpy as np

import torch

from fast_deploy.data_workload.base import Evaluator

from ..metric import *


class DetectionCOCO128Evaluator(Evaluator):

    def __init__(self, conf_thres=0.25, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.stats = []

        self.iouv = torch.linspace(0.5, 0.95,
                                   10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

        super().__init__()

    def _evaluate(self, all_outputs, all_inputs):
        pred = all_outputs[0]

        im = all_inputs[0]
        targets = all_inputs[1]
        paths = all_inputs[2]
        shapes = all_inputs[3]

        _, _, height, width = im.shape
        targets[:, 2:] *= np.array((width, height, width, height))

        self.conf_thres = 0.001
        self.iou_thres = 0.6
        pred = non_max_suppression(torch.from_numpy(pred), self.conf_thres,
                                   self.iou_thres)

        for idx, det in enumerate(pred):
            img_path = paths[idx]
            raw_img = cv2.imread(img_path)

            labels = targets[targets[:, 0] == idx, 1:]
            labels = torch.from_numpy(labels)

            nl, npr = labels.shape[0], det.shape[
                0]  # number of labels, predictions
            correct = torch.zeros(npr, self.niou, dtype=torch.bool)  # init

            if npr == 0:
                if nl:
                    self.stats.append((correct, *torch.zeros(
                        (2, 0)), labels[:, 0]))

                continue

            predn = det.clone()
            shape = shapes[idx][0]
            scale_boxes(im[idx].shape[1:], predn[:, :4], shape,
                        shapes[idx][1])  # native-space pred

            # import pdb;pdb.set_trace()
            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[idx].shape[1:], tbox, shape,
                            shapes[idx][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox),
                                    1)  # native-space labels
                correct = process_batch(predn, labelsn, self.iouv)

            self.stats.append((correct, det[:, 4], det[:, 5],
                               labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Rescale boxes from img_size to im0 size

            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                     shapes[idx][0]).round()

            # Write results
            for *box, conf, cls in reversed(det):
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                c = int(cls)  # integer class

                cv2.rectangle(raw_img,
                              p1,
                              p2,
                              colors(c, True),
                              thickness=3,
                              lineType=cv2.LINE_AA)
                label = (names[c] if False else f'{names[c]} {conf:.2f}')
                if label:
                    tf = max(3 - 1, 1)  # font thickness
                    w, h = cv2.getTextSize(
                        label, 0, fontScale=3 / 3,
                        thickness=tf)[0]  # text width, height
                    outside = p1[1] - h >= 3
                    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                    cv2.rectangle(raw_img, p1, p2, colors(c, True), -1,
                                  cv2.LINE_AA)  # filled
                    cv2.putText(
                        raw_img,
                        label,
                        (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        3 / 3, (255, 255, 255),
                        thickness=tf,
                        lineType=cv2.LINE_AA)

            save_path = img_path.replace(".jpg", "_res.jpg")
            save_path = os.path.basename(img_path).replace(".jpg", "_res.jpg")
            cv2.imwrite(save_path, raw_img)
            print(f"{save_path} saved")

    def _summary(self):
        print(f"average infer time is {self.avg_time}")

        # Compute metrics
        # import pdb;pdb.set_trace()
        stats = [torch.cat(x, 0).cpu().numpy()
                 for x in zip(*self.stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int),
                         minlength=len(names))  # number of targets per class

        print(f"mAP@.5 = {map50}, mAP@.5:.95 = {map}, mp = {mp}, mr = {mr}")
