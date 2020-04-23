import os
import sys
import random
import math
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import rocks
config = rocks.RocksConfig()
config.display()

train_dataset = rocks.RocksDataset()
valid_dataset = rocks.RocksDataset()

train_dataset.load_rocks('C:\\Users\\nemad\\PycharmProjects\\test\\Mask_RCNN\\dataset', 'train')
valid_dataset.load_rocks('C:\\Users\\nemad\\PycharmProjects\\test\\Mask_RCNN\\dataset', 'val')

train_dataset.prepare()
valid_dataset.prepare()

# ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("C:\\Users\\nemad\\PycharmProjects\\test\\Mask_RCNN\\")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# the model will be saved under ../../logs
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)



model.train(train_dataset, valid_dataset,
            learning_rate=config.LEARNING_RATE,
            epochs=60,
            layers='heads')
model.train(train_dataset, valid_dataset,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=100,
            layers="all")
class InferenceConfig(rocks.RocksConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model_pred = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
#model_path = os.path.join(ROOT_DIR, "logs/rocks20190214T2140/mask_rcnn_rocks_1000.h5")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model_pred.load_weights(model_path, by_name=True)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax
image_id = random.choice(valid_dataset.image_ids)
print(image_id)
print(valid_dataset.image_info[image_id])
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(valid_dataset, inference_config,
                           image_id, use_mini_mask=False)
if gt_mask.shape[-1] != 0:
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

image = visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            train_dataset.class_names, figsize=(8, 8))

results = model_pred.detect([original_image], verbose=1)


r = results[0]
print(r['masks'].shape)

image = visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            valid_dataset.class_names, r['scores'], ax=get_ax())

