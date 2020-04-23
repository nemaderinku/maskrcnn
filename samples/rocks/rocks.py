"""
classes for rocks dataset_old
Zhiang Chen
Dec 5, 2018
zch@asu.edu
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mrcnn import utils
from mrcnn.config import Config

#ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("C:\\Users\\nemad\\PycharmProjects\\test\\Mask_RCNN\\")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Dataset config
############################################################
class RocksConfig(Config):
    NAME = "crater"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + crater

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    GPU_COUNT = 1 # cannot create model when setting gpu count as 2

    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    # IMAGE_CHANNEL_COUNT = 1
    #
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # # IMAGE_CHANNEL = 1 # wrong, the input will be automatically converted to 3 channels (if greyscale, rgb will be repeated)
    #
    # STEPS_PER_EPOCH = 100
    # DETECTION_MIN_CONFIDENCE = 0.9
    #
    #
    # MAX_GT_INSTANCES = 512
    #
    # DETECTION_MAX_INSTANCES = 512
    #
    # TRAIN_ROIS_PER_IMAGE = 500
    # ROI_POSITIVE_RATIO = 0.5
    #
    # USE_MINI_MASK = False
    LOSS_WEIGHTS = {
        "rpn_class_loss": 0.5,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 0.5,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    

############################################################
#  Dataset
############################################################

class RocksDataset(utils.Dataset):
    def load_rocks(self, datadir, subset):
        self.add_class("crater", 1, "crater")

        # Train or validation dataset_old?
        assert subset in ["train", "val","infer"]
        dataset_dir = os.path.join('C:\\Users\\nemad\\PycharmProjects\\test\\Mask_RCNN\\dataset', subset)

        # Add images
        for filename in os.listdir(dataset_dir):
            if not filename.endswith(".npy"):
                continue


            a_image = os.path.join(dataset_dir, filename)
            # image = np.load(image_path)

            image_path = "C:\\Users\\nemad\\PycharmProjects\\test\\Mask_RCNN\\dataset\\train\\" + filename[:-4] + ".jpg"
            self.add_image(
                "crater",  ## for a single class just add the name here
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                annotation_path=a_image)

                
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info["source"] != "crater":
            return super(self.__class__, self).load_mask(image_id)
        
        mask = np.load(info["annotation_path"])
        
        if len(mask.shape) == 2:
            h,w = mask.shape
            mask_ = mask.reshape((h,w,1)).astype(np.bool)
            return mask_, np.zeros(1).astype('int32')
        
        else:
            h,w,c = mask.shape
            mask_ = np.zeros(mask.shape, dtype='uint8')
            mask_ = np.logical_or(mask, mask_)
            classes = np.ones([mask.shape[-1]], dtype=np.int32)        
            return mask_, classes
        

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "crater":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
    def display_mask(self, image_id):
        masks, ids = self.load_mask(image_id)
        mask = masks.max(2)
        plt.imshow(mask)
        plt.show()


############################################################
#  Training
############################################################

if __name__ == '__main__':
    config = RocksConfig()
    config.display()
    dataset = RocksDataset()
    dataset.load_rocks('C:\\Users\\nemad\\PycharmProjects\\test\\Mask_RCNN\\dataset', 'train')
    m, cls = dataset.load_mask(0)
    print(m[0,:,:].max())
    print(cls)
    print(dataset.image_reference(0))

    
