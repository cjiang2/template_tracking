"""
Template Tracking Python
Base Configurations class.

# References:
https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py
"""

import numpy as np
import os
import multiprocessing

class Config(object):
    """# Base configuration class. Don't use this class directly. 
    Instead, sub-class it and override the configurations you need 
    to change.
    """
    # Name the configurations
    NAME = None  # Override in sub-classes

    # Shape of bird-eye view patch
    REGION_SHAPE = (100, 100)

    # Levels of sub-templates
    L = 3
    
    # Number of sample points
    Np = 100

    # Maximum search iteration for update
    MAX_ITER = 10

    # Search mode, whether to use greedy search or Beam Search in search iteration
    SEARCH_MODE = 'greedy'

    # Beam search size
    BEAM_SIZE = 3
    
    # No. synthetic samples to be generated for each motion param
    NUM_SYNTHESIS = 500

    # sigma_d and sigma_t for motion generation
    MOTION_PARAMS = [(0.015, 0.01), 
                     (0.03, 0.02), 
                     (0.06, 0.04)]

    # Flag for debugging mode
    DEBUG = False

    def __init__(self):
        """Set values of computed attributes."""
        # No. workers for multiprocessing
        if os.name is 'nt':
            self.WORKERS = 0
        else:
            self.WORKERS = multiprocessing.cpu_count()

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")