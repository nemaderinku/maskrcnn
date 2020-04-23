"""
classes for lunar crater dataset_old
Zhiang Chen
Sep 13, 2018
zch@asu.edu
"""

import os
import sys
import numpy as np
import skimage.draw
import pickle
import argparse
import matplotlib.pyplot as plt

from mrcnn import visualize
from mrcnn.config import Config
from mrcnn import model as modellib, utils

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Dataset config
############################################################
class CraterConfig(Config):
    NAME = "crater"
    GPU_COUNT = 1 # cannot create model when setting gpu count as 2
    
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + crater
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # IMAGE_CHANNEL = 1 # wrong, the input will be automatically converted to 3 channels (if greyscale, rgb will be repeated)
    
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9
    
    
    MAX_GT_INSTANCES = 500
    
    DETECTION_MAX_INSTANCES = 600
    
    TRAIN_ROIS_PER_IMAGE = 1000
    

############################################################
#  Dataset
############################################################

class CraterDataset(utils.Dataset):
    def load_crater(self, datadir, subset, subsubset):
        self.add_class("lunar_crater", 1, "lunar_crater")
        assert subset in ["train", "val"]
        subset_dir = os.path.join(datadir, subset)
        dataset_dir = os.path.join(subset_dir, subsubset)
        annotation_path = os.path.join(dataset_dir, 'annotations.pickle')
        assert os.path.isfile(annotation_path)
        
        with open(annotation_path, "rb") as f:
            annotations = pickle.load(f, encoding='latin1')
        del(f)
        
        print('loading ' + subsubset)
        for i in range(50):
            image_path = os.path.join(dataset_dir, "img_{i:0{zp}d}.jpg".format(i=i, zp=2))
            #print(image_path)
            assert os.path.isfile(image_path)
            image_id = int(subsubset)*50 + i
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            index = "{k:0{zp}d}".format(k=i, zp=2)
            mask = annotations[index]['data']
            mask = np.swapaxes(mask, 0, 1)
            mask = np.swapaxes(mask, 1, 2)
            
            self.add_image(
                "lunar_crater",
                image_id=image_id,
                path=image_path,
                width=width, 
                height=height,
                annotation_path=annotation_path,
                annotation = mask)
        
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info["source"] != "lunar_crater":
            return super(self.__class__, self).load_mask(image_id)
        
        mask = info["annotation"]
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "lunar_crater":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
    def display_mask(self, image_id):
        masks, ids = self.load_mask(image_id)
        mask = mask.max(2)
        plt.imshow(mask)
        plt.show()
    

############################################################
#  Training
############################################################

if __name__ == '__main__':
    config = CraterConfig()
    config.display()
    dataset = CraterDataset()
    dataset.load_crater('../../dataset_old/lunar_craters', 'train', '0')
    dataset.load_crater('../../dataset_old/lunar_craters', 'train', '1')
    #dataset_old.load_crater('../../dataset_old/lunar_craters', 'train', '2')
    #dataset_old.load_crater('../../dataset_old/lunar_craters', 'train', '3')
    a,b = dataset.load_mask(65)
    
