import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import tracking utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from tracking.utils import sample_region, random_uniform_homography, normalize_minmax, apply_homography
from tracking.visualize import draw_region
from datasets.MTF import read_tracking_data

# Test of region sampling
# ------------------------------
ground_truths = read_tracking_data(ROOT_DIR+'/datasets/TMT/nl_cereal_s1.txt')
im1 = cv2.imread(ROOT_DIR+'/datasets/TMT/nl_cereal_s1/frame00001.jpg', 0)
corners_1 = ground_truths[0]

vis1 = draw_region(im1, corners_1)
plt.imshow(vis1)
plt.title('Frame 1')
plt.show()

# Sample patches, with resizing inwards, using least square
# Template patch, direct resizing
patch_1 = sample_region(im1, corners_1, region_shape=(128, 128))
patch_1 = normalize_minmax(patch_1)
print('Patch 1')
print('-'*30)
print(patch_1, patch_1.dtype)
print()

# One synthetic motion
# ------------------------------
np.random.seed(0)
current_warp = np.identity(3)  # Current warp is an identity in this case
print('Current warp:', current_warp)

# Generate synthetic motion
H, delta_p = random_uniform_homography(dist_range=3, corners=corners_1)  # NOTE: Inverse H is returned
current_warp = np.matmul(current_warp, H)  # Inverse composition
search_corners = np.round(apply_homography(current_warp, corners_1)).astype(int)
print('original corners:', corners_1)
print('search corners:', search_corners)

vis1 = draw_region(vis1, search_corners, (0, 0, 255))
plt.imshow(vis1)
plt.title('Frame 1(corners)')
plt.show()