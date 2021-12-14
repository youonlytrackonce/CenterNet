from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import sys
import cv2
import numpy as np
import pickle
import os


ANN_PATH = '/home/fatih/mnt/datasets/MyDataset/AllSeasons/set0/cam1/Annotations/coco/annotations/instances_default.json'
print(ANN_PATH)
if __name__ == '__main__':
    pred_path = '/home/fatih/mnt/datasets/MyDataset/AllSeasons/set0/cam1/CenterNet/coco/annotations/train.json'
    coco = coco.COCO(ANN_PATH)
    dets = coco.loadRes(pred_path)
    img_ids = coco.getImgIds()
    num_images = len(img_ids)
    coco_eval = COCOeval(coco, dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


