"""
Template Tracking Python
Codes for real time camera settings.
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

class CAMConfig(Config):
    """Configuration for real time camera feed.
    """
    # Give the configuration a recognizable name
    NAME = "CAM"
    LAMBD = 0.1