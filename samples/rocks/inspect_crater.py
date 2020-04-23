import rocks
import matplotlib.pyplot as plt
import os
import sys
import skimage.draw
import pickle
import argparse
import numpy as np

config = rocks.RocksConfig()
config.display()
dataset = rocks.RocksDataset()
dataset.load_rocks('../../dataset/zhiang_c3', 'train')

masks, class_ids = dataset.load_mask(2)
print(dataset.image_reference(12))
index = np.random.randint(0, class_ids.shape[0], 1)[0]
#print(index)
#print(masks.shape)
plt.imshow(masks[:,:,index])
plt.show()

image_id = 0

img = dataset.load_image(image_id)
print(img.shape)
plt.imshow(img)
plt.show()

masks, ids = dataset.load_mask(image_id)
mask = masks.max(2)
print(mask.shape)
plt.imshow(mask)
plt.show()

