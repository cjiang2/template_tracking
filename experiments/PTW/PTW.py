"""
Template Tracking Python
Codes to interact with PTW dataset.

# To acquire PTW dataset, check:
http://www.dabi.temple.edu/~hbling/data/POT-210/planar_benchmark.html

# To download template tracking datasets in MTF formats, check:
http://webdocs.cs.ualberta.ca/~vis/mtf/index.html
"""

import os
import sys

import cv2
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import tracking scripts
sys.path.append(ROOT_DIR)  # To find local version of the library
from tracking.config import Config

# ------------------------------
# Configuration
# ------------------------------

class PTWConfig(Config):
    """Configuration for inferencing on PTW dataset.
    """
    # Give the configuration a recognizable name
    NAME = "PTW"
    LAMBD = 0.1

# ------------------------------
# Functions to Process Data in MTF Format
# ------------------------------

def read_annotation_mtf(filename):
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
    
def _load_video(fpath):
    """Helper to read video file.
    """
    cap = cv2.VideoCapture()
    if not cap.open(fpath):
        raise Exception('The video file ', src_fname, ' could not be opened')
    return cap
    
def load_video_by_name(video_name, 
                       video_ftype='.avi'):
    """Helper function to read video file and the associated annotation.
    """
    cap = _load_video(os.path.join('data', video_name + video_ftype))
    gt = read_annotation_mtf(os.path.join('data', video_name + '.txt'))
    return cap, gt