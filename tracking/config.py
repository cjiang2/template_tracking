"""
Tracking Python
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

    # Patch shape to be used for tracking
    PATCH_SHAPE = (30, 30)

    # Maximum iteration
    MAX_ITER = 1

    # Debugging Flag
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