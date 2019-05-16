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

from tracking import utils

class NNTracker():
    def __init__(self, patch_shape=(50, 50), max_iter=10, Np=500, 
                 motion_params=[(0.015, 0.01), (0.03, 0.02), (0.06, 0.04)],
                 debug=True):
        """Parameter settings.
        Args:
            patch_shape: shape of bird-eye view patch.
            Np: No. synthetic data to be generated.
            motion_params: sigma_d and sigma_t for motion generation.
        """
        self.patch_shape = patch_shape
        self.max_iter = max_iter
        self.Np = Np
        self.motion_params = motion_params
        self.debug = debug

    def initialize(self, frame, corners):
        """Initialize registration tracker with tracked corners.
        """
        self.init_frame = frame
        self.warp = utils.square_to_corners_warp(corners)     # Current warp

        if self.debug:
            template = utils.sample_region(frame, corners, self.patch_shape)
            plt.imshow(template, cmap='gray')
            plt.title('template')
            plt.show()
    
        # Start synthesis now
        self.initialized = self._synthesis()

    def _synthesis(self):
        """Generate synthetic search patches and updates.
        """
        def get_patch(frame, warp, H, patch_shape):
            """Get one synthetic search patch by input homography warp.
            """
            disturbed_warp = np.matmul(self.warp, np.linalg.inv(H))   # Inverse composition, one step

            # Get search corners and search patch
            search_corners = np.round(utils.apply_homography(disturbed_warp, utils._SQUARE)).astype(int)
            search_patch = utils.sample_region(self.init_frame, search_corners, self.patch_shape)
            return np.float32(search_patch)

        print('[Running] Generating synthetic dataset...')
        self.X, self.Y = [], [] # search patches and warps
        for motion_param in self.motion_params:
            for _ in range(self.Np):
                # Random homography motion
                sigma_d, sigma_t = motion_param
                H = utils.random_homography(sigma_d, sigma_t)

                # Get search patch
                search_patch = get_patch(self.init_frame, self.warp, H, self.patch_shape)
                search_patch = (search_patch - search_patch.mean()) / search_patch.std()    # 0-1 normalization

                self.X.append(search_patch.reshape(-1))
                self.Y.append(H)
                #print('generated corners:', search_corners)

        # Add an identity warp
        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        search_patch = get_patch(self.init_frame, self.warp, H, self.patch_shape)
        search_patch = (search_patch - search_patch.mean()) / search_patch.std()

        self.X.append(search_patch.reshape(-1))
        self.Y.append(H)

        # Synthetic dataset
        self.X, self.Y = np.array(self.X), np.array(self.Y)
        print('[OK] Synthesis done.')
        print('Synthetic dataset:', self.X.shape, self.Y.shape)

        return True

    def update(self, frame):
        """Produce updated tracked region.
        """
        # np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis=1))
        if not self.initialized:
            raise Exception('Tracker uninitialized!')
        
        for i in range(self.max_iter):
            # Acquire current patch
            current_corners = np.round(utils.apply_homography(self.warp, utils._SQUARE)).astype(int)
            current_patch = utils.sample_region(frame, current_corners, self.patch_shape)
            current_patch = np.float32(current_patch)
            current_patch = (current_patch - current_patch.mean()) / current_patch.std()

            # Best match
            distances = np.sum(np.square(self.X - np.expand_dims(current_patch, axis=0).reshape(-1)), axis=1)
            #print(distances.shape)
            idx = np.argmin(distances)

            # Update tracker
            update_warp = self.Y[idx]
            self.warp = np.matmul(self.warp, update_warp)
            self.warp = utils.normalize_hom(self.warp)

        return np.round(utils.apply_homography(self.warp, utils._SQUARE)).astype(int)
