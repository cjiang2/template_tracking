import os
import sys

import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import tracking utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from tracking import utils

def draw_region(img, corners, color=(0, 255, 0), thickness=2):
    """
    Draw the bounding box specified by the given corners
    of shape (2, 4).
    corners: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] four rectangle 
             corner coordinates.
    """
    if len(img.shape) < 3:
        vis = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    corners = np.round(corners).astype(int) # Force int
    for i in range(4):
        p1 = (corners[0, i], corners[1, i])
        p2 = (corners[0, (i + 1) % 4], corners[1, (i + 1) % 4])
        cv2.line(vis, p1, p2, color, thickness)
    return vis