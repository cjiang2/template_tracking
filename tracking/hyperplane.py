"""
Hyperplane Tracker
The main implementation for Hyperplane tracker.

# References:
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import linear_model

from tracking import utils

class HyperplaneTracker():
    def __init__(self, 
                 patch_shape=(30, 30), 
                 N=5000, 
                 motion_params=[(0.12, 0.08),
                                (0.09, 0.06),
                                (0.06, 0.04), 
                                (0.03, 0.02), 
                                (0.015, 0.01)], 
                 max_iter=10,
                 lambd=0.1,
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
        self.lambd = lambd
        self.debug = debug

    def initialize(self, 
                   frame, 
                   corners):
        """Initialize registration tracker with tracked corners.
        """
        self.init_frame = frame
        self.warp = utils.square_to_corners_warp(corners)     # Current warp

        self.template = utils.sample_region(frame, corners, self.patch_shape)
        self.template = np.float32(self.template)
        self.template = utils.normalize_minmax(self.template)

        if self.debug:
            template = utils.sample_region(frame, corners, self.patch_shape)    # Visualization test
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
        for _ in range(len(self.motion_params)):
            self.X.append([])
            self.Y.append([])

        for i, motion_param in enumerate(self.motion_params):
            for _ in range(self.N):
                # Random homography motion
                sigma_d, sigma_t = motion_param
                H = utils.random_homography(sigma_d, sigma_t)

                # Get search patch
                search_patch = get_patch(self.init_frame, self.warp, H, self.patch_shape)
                search_patch = utils.normalize_minmax(search_patch)

                # Image Difference
                deltaI = (search_patch - self.template).reshape(-1)

                H = H.reshape(-1)
                self.X[i].append(deltaI)
                self.Y[i].append(H)

        for i in range(len(self.X)):
            self.X[i], self.Y[i] = np.array(self.X[i]), np.array(self.Y[i])

        # Synthetic dataset
        print('[OK] Synthesis done.')
        for i in range(len(self.X)):
            print('Synthetic dataset:', self.X[i].shape, self.Y[i].shape)
        
        # Train now
        self.train()

        return True

    def train(self):
        """Use an Scipy linear regression instead.
        """
        self.learners = []
        # Calculate weight matrix per motion param
        for i in range(len(self.motion_params)):
            learner = linear_model.Ridge(alpha=self.lambd).fit(self.X[i], self.Y[i])
            self.learners.append(learner)

    def update(self, 
               frame):
        """Produce updated tracked region.
        """
        if not self.initialized:
            raise Exception('Tracker uninitialized!')

        for _ in range(self.max_iter):
            # Acquire current patch
            current_corners = np.round(utils.apply_homography(self.warp, utils._SQUARE)).astype(int)
            current_patch = utils.sample_region(frame, current_corners, self.patch_shape)
            current_patch = np.float32(current_patch)
            current_patch = utils.normalize_minmax(current_patch)

            # Linear regression
            deltaI = np.expand_dims((current_patch - self.template).reshape(-1), axis=0)

            # Greedy search
            #TODO: can do a beamsearch here
            scores = []
            candidates = []
            for learner in self.learners:
                # Produce updating candidates
                update_warp = learner.predict(deltaI).squeeze(0)
                update_warp = update_warp.reshape(3, 3)

                # Candidate updated warp
                candidate_warp = np.matmul(self.warp, update_warp)
                candidate_warp = utils.normalize_hom(candidate_warp)

                # Get candidate patch
                candidate_corners = np.round(utils.apply_homography(candidate_warp, utils._SQUARE)).astype(int)
                candidate_patch = utils.sample_region(frame, candidate_corners, self.patch_shape)
                candidate_patch = np.float32(candidate_patch)
                candidate_patch = utils.normalize_minmax(candidate_patch)

                # Image similarity score
                score = np.sum(np.square(candidate_patch - self.template))
                scores.append(score)
                candidates.append(update_warp)

            # Get minimum score
            idx = scores.index(min(scores))
            update_warp = candidates[idx]

            # Update
            self.warp = np.matmul(self.warp, update_warp)
            self.warp = utils.normalize_hom(self.warp)

        return np.round(utils.apply_homography(self.warp, utils._SQUARE)).astype(int)