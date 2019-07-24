"""
Template Tracking Python
Codes to interact with datasets in MTF format.

# To download template tracking datasets in MTF formats, check:
http://webdocs.cs.ualberta.ca/~vis/mtf/index.html

"""

import os
import sys

import cv2
import numpy as np

def read_tracking_data(filename):
    """Read ground truth data in MTF format.
    """
    if not os.path.isfile(filename):
        print("Tracking data file not found:\n ", filename)
        sys.exit()

    data_file = open(filename, 'r')
    data_array = []
    line_id = 0
    for line in data_file:
        if line_id > 0:
            line = line.strip().split()
            coordinate = np.array([[float(line[1]), float(line[3]), float(line[5]), float(line[7])],
                                [float(line[2]), float(line[4]), float(line[6]), float(line[8])]])
            coordinate = np.round(coordinate)
            data_array.append(coordinate.astype(int))
        line_id += 1
    data_file.close()
    return np.array(data_array)

def read_imgs_folders(root_dir,
                      folder_name, 
                      video_name):
    """Read tracking dataset stored as sequences of images.
    """
    src_fname = os.path.join(root_dir, 'datasets', folder_name, video_name, 'frame%05d.jpg')
    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        raise Exception('The video file ', src_fname, ' could not be opened')

    ground_truths = read_tracking_data(os.path.join(root_dir, 'datasets', folder_name, video_name+'.txt'))
    return cap, ground_truths

def read_video_folders(folder_name, 
                       video_name,
                       video_type='.avi'):
    """Read tracking dataset stored as video file.
    """
    src_fname = os.path.join(ROOT_DIR, 'datasets', folder_name, video_name + video_type)
    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        raise Exception('The video file ', src_fname, ' could not be opened')

    ground_truths = read_tracking_data(os.path.join(ROOT_DIR, 'datasets', folder_name, video_name+'.txt'))
    return cap, ground_truths