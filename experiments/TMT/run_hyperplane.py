import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import tracking utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from tracking.hyperplane import HyperplaneTracker
from tracking.visualize import draw_region
from tracking.config import Config
from datasets.MTF import read_tracking_data, read_imgs_folders

def alignment_error(corners_pred, 
                    corners_true):
    """Calculate Alignment Error (l2) error between corners.
    """
    return np.sqrt(np.mean(np.sum((corners_pred - corners_true)**2, axis=0)))

def run_hyperplane_tracker(config, 
                           cap, 
                           ground_truths):
    """Helper function to run hyperplane tracker.
    """
    # Prepare 1st frame
    ret, frame0_rgb = cap.read()
    frame0 = cv2.cvtColor(frame0_rgb, cv2.COLOR_BGR2GRAY)
    corners0 = ground_truths[0,:]
    if config.DEBUG:
        print(corners0, corners0.shape)

    # Initialize nearest neighbor tracker
    tracker = HyperplaneTracker(patch_shape=config.PATCH_SHAPE,
                                N=config.NUM_SYNTHESIS,
                                max_iter=config.MAX_ITER,
                                debug=config.DEBUG)
    tracker.initialize(frame0, corners0)

    # Visualization
    window_name = 'Tracking Result'
    cv2.namedWindow(window_name)
    errors = []

    # Track
    for i in range(1, ground_truths.shape[0]):
        ret, frame = cap.read()
        if not ret:
            print('Frame ", i, " could not be read')
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = tracker.update(frame_gray)
        err = alignment_error(corners, ground_truths[i,:])
        errors.append(err)
        print('Frame id: {}, Mean Corners Error: {}\nCorners: {}'.format(i, err, corners))

        # Drawings and visualization here
        frame = draw_region(frame, ground_truths[i,:])
        frame = draw_region(frame, corners, color=(0, 0, 255))
        cv2.imshow(window_name, frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    print('\nSummary')
    print('-'*20)
    print('Average Mean Corners Error: {}'.format(sum(errors)/len(errors)))

    return

# Sub-config class for hperparameters
class HyperplaneConfig(Config):
    """Configuration for hyperplane tracker.
    """
    NAME = 'HyperplaneTracker'
    MAX_ITER = 10    # Max iteration per H
    NUM_SYNTHESIS = 1000
    DEBUG = True

if __name__ == '__main__':
    # Parameters
    folder_name = 'PAMI'
    video_name = 'box'
    hyperplane_config = HyperplaneConfig()

    # Experiment
    cap, ground_truths = read_imgs_folders(ROOT_DIR, folder_name, video_name)
    run_hyperplane_tracker(hyperplane_config, cap, ground_truths)