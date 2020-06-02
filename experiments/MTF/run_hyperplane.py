"""
Template Tracking Python
Run experiment with the hyperplane tracker.
"""

import os
import sys

import cv2
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import tracking utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from tracking import utils
from tracking.hyperplane import HyperplaneTracker
from tracking import visualize
from experiments.MTF import MTF

# Some flags, to be moved into config class
DATASET_NAME = 'PTW'
VIDEO_NAME = 'Coke_2'
READ_FOLDER = False

def alignment_error(corners_pred, 
                    corners_true):
    """Calculate Alignment Error (l2) error between corners.
    """
    return np.mean(np.sqrt(np.sum((corners_pred - corners_true)**2, axis=0)))

def run_hyperplane_tracker(config, 
                           cap, 
                           gt):
    """Helper function to run hyperplane tracker.
    """
    # Prepare 1st frame
    ret, frame0_rgb = cap.read()
    frame0 = cv2.cvtColor(frame0_rgb, cv2.COLOR_BGR2GRAY)
    corners0 = gt[0,:]

    # Initialize hyperplane tracker
    tracker = HyperplaneTracker(config)
    tracker.initialize(frame0, corners0)

    # Visualization
    window_name = 'Tracking Result'
    cv2.namedWindow(window_name)
    errors = []

    # Track
    for i in range(1, gt.shape[0]):
        ret, frame = cap.read()
        if not ret:
            print('Frame ", i, " could not be read')
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = tracker.update(frame_gray)
        err = alignment_error(corners, gt[i,:])
        errors.append(err)
        print('Frame id: {}, Mean Corners Error: {}\nCorners: {}'.format(i, err, corners))

        # Drawings and visualization here
        frame = visualize.draw_region(frame, gt[i,:])
        frame = visualize.draw_region(frame, corners, color=(0, 0, 255))
        cv2.imshow(window_name, frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    print('\nSummary')
    print('-'*20)
    print('Average Mean Corners Error: {}'.format(sum(errors)/len(errors)))

    if config.DEBUG:
        # Print out some trajectories
        print('No. trajectories recorded: {}'.format(len(tracker._trajectories)))
        traj_0 = tracker._trajectories[0]
        print(traj_0.warps, traj_0.score)

        traj_rand = tracker._trajectories[12]
        print(len(traj_rand.warps), traj_rand.score)

    return

if __name__ == '__main__':
    # Configuration
    config = MTF.MTFConfig()
    
    # Load the video to experiment on
    if READ_FOLDER:
        cap, gt = MTF.load_video_by_folder(DATASET_NAME, VIDEO_NAME)
    else:
        cap, gt = MTF.load_video_by_name(DATASET_NAME, VIDEO_NAME)
    
    # Run tracker now
    run_hyperplane_tracker(config, cap, gt)