import os
import sys

import numpy as np
import cv2

# ------------------------------------------------------------
# Functions for sampling, homogeneous
# ------------------------------------------------------------

def sample_region(img, corners, region_shape=None,
                  method=0, interpolation=cv2.INTER_LINEAR):
    """
    Sample corners region as Bird-eye view patch. Left top point of the 
    rect corners is defined as the origin coordinate (0, 0).
    Args:
        img: image to be sampled
        corners: (2, 4) coordinates.
        region_shape: (height, width) of the final sampled region.
        method: 0, cv2.RANSAC or other options here.
        interpolation: interpolation for perspective transformation.
    Returns:
        region: sampled img region.
    """
    if region_shape is None:
        # Top-left, Top-right, Bottom-right, Bottom-left coordinates
        tl, tr, br, bl = corners.T
        # Calculate widths and heights of the corners rect
        width_1 = np.sqrt(np.square(tl[0] - tr[0]) + np.square(tl[1] - tr[1]))
        width_2 = np.sqrt(np.square(bl[0] - br[0]) + np.square(bl[1] - br[1]))
        height_1 = np.sqrt(np.square(tl[0] - bl[0]) + np.square(tl[1] - bl[1]))
        height_2 = np.sqrt(np.square(tr[0] - br[0]) + np.square(tr[1] - br[1]))

        # Choose the maximum width and height
        width = int(np.round(max(width_1, width_2)))
        height = int(np.round(max(height_1, height_2)))
    else:
        height, width = region_shape

    # Homography for image warping
    # Origin coordinate
    src = np.float32(corners.T)
    dst = np.array([[0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]], dtype=np.float32)
    dst[dst < 0] = 0

    # Birds-eye-view for the patch image based on corners
    M, _ = cv2.findHomography(src, dst, method)   # OpenCV accepts (4, 2) only
    region = cv2.warpPerspective(img, M, (width, height), flags=interpolation)

    return region