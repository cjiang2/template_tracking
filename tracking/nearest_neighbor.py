"""
Nearest Neighbor Tracker
The main implementation for Nearest Neighbor registration-based tracker.

# References:
http://www.roboticsproceedings.org/rss09/p44.pdf
https://github.com/abhineet123/PTF/tree/master
"""
from typing import Tuple

import pyflann
import numpy as np
import cv2
import matplotlib.pyplot as plt

from .utils import *

class NearestNeighbor(object):
    def __init__(
        self,
        max_iter: int = 1,
        distance_type: str = "euclidean",
        region_resolution: Tuple[int] = (50, 50),
        motion_params: Tuple[Tuple[float]] = [(0.06, 0.04), (0.03, 0.02), (0.015, 0.01)], 
        n_samples: int = 500,
        update_template: bool = False,
        ):
        self.max_iter = max_iter
        self.distance_type = distance_type
        self.region_resolution = region_resolution
        self.motion_params = motion_params
        self.n_samples = n_samples
        self.update_template = update_template

    def quadrilateral_init(
        self, 
        img: np.ndarray, 
        corner: np.ndarray,
        ):
        """Initialize with a quadrilateral.
        """
        self.img = img
        self.warp = square_to_corner_warp(corner)     # Current warp
        self.trajectory = []    # Keep track of updating trajectories

        # Generate lookup table
        self.synthesize()

    def synthesize(self):
        """Synthesize lookup table for candidate warps.
        """
        candidates, self.P = [], []

        for (sigma_d, sigma_t) in self.motion_params:
            for _ in range(self.n_samples):
                p = generate_random_warp(sigma_d, sigma_t)
                disturbed_warp = normalize_hom(np.matmul(self.warp.copy(), np.linalg.inv(p)))     # Inverse compositional
                candidate = get_birdeye_view_(self.img, disturbed_warp, self.region_resolution)
                candidates.append(candidate.reshape(-1))
                self.P.append(p)
            # plt.imshow(candidate)
            # plt.show()

        # Add an identity warp
        p = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        disturbed_warp = normalize_hom(np.matmul(self.warp.copy(), np.linalg.inv(p)))     # Inverse compositional
        candidate = get_birdeye_view_(self.img, disturbed_warp, self.region_resolution)
        candidates.append(candidate.reshape(-1))
        self.P.append(p)

        self.flann = pyflann.FLANN()
        self.flann.build_index(np.array(candidates), algorithm='kdtree', trees=6, checks=50)

    def update(
        self, 
        img: np.ndarray,
        ):
        """Produce updated tracked region.
        """
        for i in range(self.max_iter):
            # Search
            search = get_birdeye_view_(img, self.warp, self.region_resolution)

            # Best match
            results, dists = self.flann.nn_index(search.reshape(-1))
            p = self.P[int(results[0])]     # 1-NN
            self.warp = normalize_hom(np.matmul(self.warp, p))
            
            # Store trajectory
            self.trajectory.append(p)

        # Update the template
        if self.update_template:
            self.template = get_birdeye_view_(img, self.warp, self.region_resolution)

        return self.warp