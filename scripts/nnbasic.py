import os
import sys
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../")
sys.path.append(os.path.join(ROOT_DIR))  # To find local version of the library

from tracking.utils import *
from tracking.nearest_neighbor import NearestNeighbor

def draw_region(img, corners, color=(0, 255, 0), thickness=2):
    """Draw the bounding box specified by the given corners
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default=None, 
                        help='Directory of the video.')
    args = parser.parse_args()

    cap, gt = load_quadrilateral_video(args.video_dir)

    # Work with 1st frame
    _, img0 = cap.read()
    corner0 = gt[0, :]

    tracker = NearestNeighbor()
    tracker.quadrilateral_init(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), corner0)

    window_name = 'Tracking'
    cv2.namedWindow(window_name)
    errors = []

    # Track
    for i in range(1, gt.shape[0]):
        ret, frame = cap.read()
        if not ret:
            print('Frame ", i, " could not be read')
            break
        
        warp = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        corner = np.round(corner_to_square(warp)).astype(int)

        # Drawings and visualization here
        vis = frame.copy()
        vis = draw_region(vis, gt[i,:])
        vis = draw_region(vis, corner, color=(0, 0, 255))
        cv2.imshow(window_name, vis)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break