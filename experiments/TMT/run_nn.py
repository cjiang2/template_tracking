import os
import sys

import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import tracking utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from tracking.nearest_neighbor import NNTracker
from tracking.visualize import draw_region
from datasets.TMT import read_tracking_data

def alignment_error(corners_pred, corners_true):
    """Calculate Alignment Error (l1) error between corners.
    """
    return np.sum(np.absolute(corners_pred - corners_true) / 4)

def read_TMT(video_name):
    """Read TMT video information.
    """
    src_fname = os.path.join(ROOT_DIR, 'datasets', 'TMT', video_name, 'frame%05d.jpg')
    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        raise Exception('The video file ', src_fname, ' could not be opened')

    ground_truths = read_tracking_data(os.path.join(ROOT_DIR, 'datasets', 'TMT', video_name+'.txt'))
    return cap, ground_truths

def run_nn_tracker(cap, ground_truths, debug=True):
    # Prepare 1st frame
    ret, frame0_rgb = cap.read()
    frame0 = cv2.cvtColor(frame0_rgb, cv2.COLOR_BGR2GRAY)
    corners0 = ground_truths[0,:]
    if debug:
        print(corners0, corners0.shape)

    # Initialize nearest neighbor tracker
    nntracker = NNTracker()
    nntracker.initialize(frame0, corners0)

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
        corners = nntracker.update(frame_gray)
        err = alignment_error(corners, ground_truths[i,:])
        errors.append(err)
        print('Frame id: {}, Mean Corners Error: {}, Corners: {}'.format(i, err, corners))

        # Drawings and visualization here
        frame = draw_region(frame, ground_truths[i,:])
        frame = draw_region(frame, corners, color=(255, 0, 0))
        cv2.imshow(window_name, frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    print('\nSummary')
    print('-'*20)
    print('Average Mean Corners Error: {}'.format(sum(errors)/len(errors)))

    return

if __name__ == '__main__':
    # Parameters
    video_name = 'dl_bookI_s5'

    # Experiment
    cap, ground_truths = read_TMT(video_name)
    run_nn_tracker(cap, ground_truths)