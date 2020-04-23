import os
import sys
import random
import math
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import rocks

config = rocks.RocksConfig()
config.display()

valid_dataset = rocks.RocksDataset()
valid_dataset.load_rocks('C:\\Users\\nemad\\PycharmProjects\\test\\Mask_RCNN\\dataset', 'infer')

valid_dataset.prepare()

ROOT_DIR = os.path.abspath("../.././")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
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
model_path = "C:\\Users\\nemad\\PycharmProjects\\test\\craters\\logs\\crater20200420T2028\\mask_rcnn_crater_0008.h5"#"C:\\Users\\nemad\\PycharmProjects\\test\\Mask_RCNN\\logs\\crater20200421T1235\\mask_rcnn_crater_0100.h5" #os.path.join(ROOT_DIR, "logs\\crater20200327T1511\\mask_rcnn_crater_0005.h5")
# model_path = model_pred.find_last()

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


instances = list()

for info in valid_dataset.image_info:
    image_id = info['id']
    # print("---------------------------------------------", image_id)
    image = cv2.imread(info['path'])
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(valid_dataset, inference_config,
                               image_id, use_mini_mask=False)

    results = model_pred.detect([image], verbose=1)
    r = results[0]
    print(r['masks'].shape)

    ### display original image
    # original_image = visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, valid_dataset.class_names, figsize=(8, 8))
    ### display inference image
    # image = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], valid_dataset.class_names, r['scores'], ax=get_ax())
    name = 'masked_' + info['path'].split('/')[-1]
    ### save inference image
    # cv2.imwrite("../../dataset/zhiang_c3/infer/"+name, image)

    ### get all instances
    image_name = info['path'].split('/')[-1]
    coord_str = image_name.split('.')[0]
    coord = coord_str.split('_')
    coord = [int(coord[0]), int(coord[1])]

    for i, bb in enumerate(r['rois']):
        instance = dict()
        instance['coord'] = coord
        instance['bb'] = bb
        mask = r['masks'][:, :, i].astype(float)
        mask = cv2.resize(mask, (400, 400)).astype(bool)
        instance['mask'] = mask
        instances.append(instance)
    print(len(instances))
    print('*' * 20)

i = np.random.randint(len(instances))
print(instances[i]['bb'])
mask = instances[i]['mask']
plt.imshow(mask)
plt.show()


l = len(instances)
for i in range(0,l,10000):
    if l-i>10000:
        s = instances[i:i+10000]
    else:
        s = instances[i:]
    with open('instances_'+str(int(i/10000))+'.pickle', 'wb') as f:
        pickle.dump(s, f, protocol=pickle.HIGHEST_PROTOCOL)
