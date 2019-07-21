"""
Nearest Neighbor Tracker
The main implementation for Nearest Neighbor registration-based tracker.

# References:
http://www.roboticsproceedings.org/rss09/p44.pdf
https://github.com/abhineet123/PTF/tree/master
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyflann

from tracking import utils

class NNTracker():
    def __init__(self, 
                 patch_shape=(50, 50), 
                 max_iter=10, 
                 N=500, 
                 motion_params=[(0.12, 0.08),
                                (0.09, 0.06),
                                (0.06, 0.04), 
                                (0.03, 0.02), 
                                (0.015, 0.01)], 
                 debug=False):
        """Parameter settings.
        Args:
            patch_shape: shape of bird-eye view patch.
            N: No. synthetic data to be generated.
            motion_params: sigma_d and sigma_t for motion generation.
        """
        self.patch_shape = patch_shape
        self.max_iter = max_iter
        self.N = N
        self.motion_params = motion_params

        # Debugging option
        self.debug = debug
        if self.debug:
            self._trajectories = []  # Option to store all updating warp trajectories
            self._all_corners = []   # Option to store all corners produced during tracking, including iterations

    def initialize(self, 
                   frame, 
                   corners):
        """Initialize registration tracker with tracked corners.
        """
        self.init_frame = frame
        self.warp = utils.square_to_corners_warp(corners)     # Current warp

        if self.debug:
            self._trajectories.append([self.warp])   # Store initial warp
            self._all_corners.append([corners])      # Store initial corners
            template = utils.sample_region(frame, corners, self.patch_shape)    # Visualization test
            plt.imshow(template, cmap='gray')
            plt.title('template')
            plt.show()
    
        # Start synthesis now
        self.initialized = self._synthesis()

    def _synthesis(self):
        """Generate synthetic search patches and updates.
        """
        def get_patch(frame, 
                      warp, 
                      H, 
                      patch_shape):
            """Get one synthetic search patch by input homography warp.
            """
            disturbed_warp = np.matmul(warp, np.linalg.inv(H))   # Inverse warp

            # Get search corners and search patch
            search_corners = np.round(utils.apply_homography(disturbed_warp, utils._SQUARE)).astype(int)
            search_patch = utils.sample_region(frame, search_corners, patch_shape)
            return np.float32(search_patch)

        print('[Running] Generating synthetic dataset...')
        self.X, self.Y = [], [] # search patches and warps
        for motion_param in self.motion_params:
            for _ in range(self.N):
                # Random homography motion
                sigma_d, sigma_t = motion_param
                H = utils.random_homography(sigma_d, sigma_t)

                # Get search patch
                search_patch = get_patch(self.init_frame, self.warp, H, self.patch_shape)
                search_patch = utils.normalize_minmax(search_patch)

                self.X.append(search_patch.reshape(-1))
                self.Y.append(H)

        # Add an identity warp
        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        search_patch = get_patch(self.init_frame, self.warp, H, self.patch_shape)
        search_patch = utils.normalize_minmax(search_patch)

        self.X.append(search_patch.reshape(-1))
        self.Y.append(H)

        # Synthetic dataset
        self.X, self.Y = np.array(self.X), np.array(self.Y)
        print('[OK] Synthesis done.')
        print('Synthetic dataset:', self.X.shape, self.Y.shape)

        # pyflann initialization
        print('Building pyflann indexing...')
        pyflann.set_distance_type('manhattan')
        self.flann = pyflann.FLANN()
        self.flann.build_index(self.X, algorithm='kdtree', trees=6, checks=50)
        print('[OK] done.')

        return True

    def update(self, 
               frame):
        """Produce updated tracked region.
        """
        if not self.initialized:
            raise Exception('Tracker uninitialized!')
        
        if self.debug:
            temp_corners = []
            temp_trajectories = []

        for i in range(self.max_iter):
            # Acquire current patch
            current_corners = np.round(utils.apply_homography(self.warp, utils._SQUARE)).astype(int)
            current_patch = utils.sample_region(frame, current_corners, self.patch_shape)
            current_patch = np.float32(current_patch)
            current_patch = utils.normalize_minmax(current_patch)

            # Best match
            results, dists = self.flann.nn_index(current_patch.reshape(-1))
            update_warp = self.Y[int(results[0])]
            self.warp = np.matmul(self.warp, update_warp)
            self.warp = utils.normalize_hom(self.warp)
            
            # Debugging option: store current trajectory and corners
            if self.debug:
                temp_corners.append(np.round(utils.apply_homography(self.warp, utils._SQUARE)).astype(int))
                temp_trajectories.append(update_warp)
        
        if self.debug:
            # Store corners and trajectories generated during iteration
            self._all_corners.append(temp_corners)
            self._trajectories.append(temp_trajectories)

        return np.round(utils.apply_homography(self.warp, utils._SQUARE)).astype(int)
