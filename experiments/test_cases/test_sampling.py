import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import tracking utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from tracking.utils import sample_region
from tracking.visualize import draw_region
from datasets.MTF import read_tracking_data

# Test of region sampling
# ------------------------------
ground_truths = read_tracking_data(ROOT_DIR+'/datasets/TMT/nl_cereal_s1.txt')
im1 = cv2.imread(ROOT_DIR+'/datasets/TMT/nl_cereal_s1/frame00001.jpg', 0)
im2 = cv2.imread(ROOT_DIR+'/datasets/TMT/nl_cereal_s1/frame00002.jpg', 0)
corners_1 = ground_truths[0]
corners_2 = ground_truths[1]

vis1 = draw_region(im1, corners_1)
vis2 = draw_region(im2, corners_2)

plt.imshow(vis1)
plt.title('Frame 1')
plt.show()

plt.imshow(vis2)
plt.title('Frame 2')
plt.show()

# Sample patches, with resizing inwards, using least square
# Template patch, direct resizing
patch_1 = sample_region(im1, corners_1, region_shape=(128, 128))
patch_1 = (patch_1 - np.mean(patch_1))/patch_1.std()
print('Patch 1')
print('-'*30)
print(patch_1, patch_1.dtype)
print()

# Patch after disturbance
patch_2 = sample_region(im2, corners_2, region_shape=(128, 128))
patch_2 = (patch_2 - np.mean(patch_2))/patch_2.std()
print('Patch 2')
print('-'*30)
print(patch_2, patch_2.dtype)
print()

# Image difference
im_diff = cv2.subtract(patch_1, patch_2)
im_diff_float = np.float32(patch_1) - np.float32(patch_2)

print('OpenCV Image Difference\nMax. Pixel Value: {}, Min Pixel Value: {}'.format(np.max(im_diff), np.min(im_diff)))
print('2-norm:', np.linalg.norm(im_diff))
print()

print('numpy(float) Image Difference\nMax. Pixel Value: {}, Min Pixel Value: {}'.format(np.max(im_diff_float), np.min(im_diff_float)))
print('2-norm:', np.linalg.norm(im_diff_float))

# Show a bunch of pics
plt.subplot(1,2,1)
plt.title('Patch at Frame 1')
plt.imshow(patch_1, cmap='gray')
plt.subplot(1,2,2)
plt.title('Patch at Frame 2')
plt.imshow(patch_2, cmap='gray')
plt.show()

plt.subplot(1,2,1)
plt.imshow(im_diff, cmap='gray')
plt.title('Image Difference(OpenCV)')
plt.subplot(1,2,2)
plt.imshow(im_diff_float, cmap='gray')
plt.title('Image Difference(numpy)')
plt.show()

# Test Np
patch_1 = sample_region(im1, corners_1, region_shape=(100, 100), Np=400)
plt.title('Patch at Frame 1')
plt.imshow(patch_1, cmap='gray')
plt.show()