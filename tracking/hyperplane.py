"""
Hyperplane Tracker
The main implementation for Hyperplane tracker.

# References:
https://pdfs.semanticscholar.org/f06d/3fc49dca2e6380969b3d8f377b33c6001e7a.pdf
https://pdfs.semanticscholar.org/7fbc/4c4f01eb9716959ffef8b4a620a3d1c38577.pdf
http://www.roboticsproceedings.org/rss09/p44.pdf
For beamsearch: https://github.com/tensorflow/models/blob/master/research/im2txt/im2txt/inference_utils/caption_generator.py
"""

import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import linear_model
from tracking import utils
from tracking import visualize

class Trajectory(object):
    """Represent a trajectory of warps throughout one or 
    multiple updating iterations.
    """
    def __init__(self, warps, score):
        self.warps = warps
        self.score = score

    def __lt__(self, other):
        assert isinstance(other, Trajectory)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, Trajectory)
        return self.score == other.score

    def add_warp(self, warp, score):
        self.warps.append(warp)
        self.score += score

    def __add__(self, other):
        assert isinstance(other, Trajectory)
        warps = self.warps + other.warps
        score = self.score + other.score
        return Trajectory(warps, score)

class HyperplaneTracker():
    """Main class for hyperplane tracker.
    """
    def __init__(self, 
                 config):
        self.config = config        # Store configuration
        self.initialized = False    # Flags for tracker status
        self._trajectories = [] # Option to store all updating warp trajectories

    def initialize(self, 
                   frame, 
                   corners):
        """Initialize template for hyperplane tracker.
        """
        self.frame = frame
        self.warp = utils.square_to_corners_warp(corners)     # Current warp

        # Sample full template patch
        self.template = utils.sample_region(frame, 
                                            corners, 
                                            region_shape=self.config.REGION_SHAPE,
                                            Np=self.config.Np)
        self.template = utils.normalize_minmax(np.float32(self.template))

        # Record the initial trajectory
        traj = Trajectory(warps=[self.warp], score=0.0)
        self._trajectories.append(traj)

        # DEBUG: Some visualization here
        if self.config.DEBUG:
            print('[DEBUG] corners:\n', corners)
            print('[DEBUG] warp:\n', self.warp)

            # Visualize template patch
            template_full = utils.sample_region(frame, 
                                                corners, 
                                                region_shape=self.config.REGION_SHAPE)
            template_full = utils.normalize_minmax(np.float32(template_full))

            plt.subplot(1,2,1)
            plt.imshow(self.template, cmap='gray')
            plt.title('Template\n' + str(self.template.shape[:2]))
            plt.subplot(1,2,2)
            plt.imshow(template_full, cmap='gray')
            plt.title('Template w/o subsampling\n'+ str(template_full.shape[:2]))
            plt.show()

        # Start synthesis now
        self.initialized = self._synthesis()

    def _synthesis(self):
        """Generate synthetic samples.
        """
        self.X, self.Y = [], []
        for motion_param in self.config.MOTION_PARAMS:
            sigma_d, sigma_t = motion_param
            warps = np.zeros((self.config.NUM_SYNTHESIS, 3, 3), dtype=np.float32)
            patches = np.zeros((self.config.NUM_SYNTHESIS, self.template.shape[0], self.template.shape[1]), dtype=np.float32)
            print('Generating {} synthetic samples...'.format(self.config.NUM_SYNTHESIS))
        
            for i in range(self.config.NUM_SYNTHESIS):
                # Generate random warp
                H = utils.random_hom(sigma_d, sigma_t)
                warps[i,:,:] = H
                disturbed_warp = np.matmul(self.warp, np.linalg.inv(H))     # Inverse warp
                disturbed_warp = utils.normalize_hom(disturbed_warp)

                # Grab the disturbed template
                disturbed_template = self._get_region(self.frame, disturbed_warp)
                patches[i,:,:] = disturbed_template

            # Prepare synthetic samples for learning
            X = (patches - np.expand_dims(self.template, axis=0)).reshape(-1, self.config.Np)
            self.X.append(X)
            self.Y.append(warps.reshape(-1, 9)[:,:-1])

            # DEBUG: Visualize the randomly generated patch
            if self.config.DEBUG:
                plt.subplot(1,2,1)
                plt.imshow(self.template, cmap='gray')
                plt.title('Original Template')
                plt.subplot(1,2,2)       
                plt.imshow(disturbed_template, cmap='gray')
                plt.title('Disturbed with Param: {}'.format(motion_param))
                plt.show()
        
        print('Synthesis done.')

        # Train now
        self._train()

        return True

    def _train(self):
        """Ridge linear regression. 
        Construct individual learners for:
            - Small to large motion distributions.
        """
        self.learners = []
        # Calculate weight matrix per motion param
        for i in range(len(self.X)):
            learner = linear_model.Ridge(alpha=self.config.LAMBD).fit(self.X[i], self.Y[i])
            self.learners.append(learner)
        print('Training done.')
        return

    def _get_region(self, 
                    frame,
                    warp):
        """Wrapper to acquire a region given a warp.
        """
        corners = np.round(utils.apply_to_pts(warp, utils._SQUARE)).astype(int)
        patch = utils.sample_region(frame, 
                                    corners,
                                    region_shape=self.config.REGION_SHAPE,
                                    Np=self.config.Np)
        patch = utils.normalize_minmax(np.float32(patch))
        return patch

    def _greedy_search(self, 
                       frame):
        """Greedy search method.
        """
        # Record the starting score
        traj = Trajectory(warps=[], score=0.0)

        for _ in range(self.config.MAX_ITER):
            scores = []
            candidates = []

            # Acquire current patch
            curr_patch = self._get_region(frame, self.warp)
            deltaI = np.expand_dims((curr_patch - self.template).reshape(-1), axis=0)
            for learner in self.learners:
                # Produce p to update current warp
                p = learner.predict(deltaI).squeeze(0)
                update_warp = utils.make_hom(p)

                # Candidate updated warp
                candidate_warp = np.matmul(self.warp, update_warp)
                candidate_warp = utils.normalize_hom(candidate_warp)

                # Get candidate patch
                candidate_patch = self._get_region(frame, candidate_warp)

                # Image similarity score
                score = utils.SSD(candidate_patch, self.template)
                scores.append(score)
                candidates.append(update_warp)

            # Get minimum score among updates produced by all learners
            idx = np.argmin(scores)
            update_warp = candidates[idx]
            update_score = scores[idx]

            # Greedy update
            self.warp = np.matmul(self.warp, update_warp)
            self.warp = utils.normalize_hom(self.warp)
            traj.add_warp(self.warp, update_score)

        return traj

    def _beam_search(self, 
                     frame):
        """Beam search method among iteration.
        """
        def _extract_topk(alist, k):
            """Extract the top K elements.
            """
            scores = [t.score for t in alist]
            indices = np.argsort(alist)[:k]
            return [alist[idx] for idx in indices]

        initial_beam = Trajectory(warps=[self.warp], score=0.0)
        partial_trajs = []
        partial_trajs.append(initial_beam)
        complete_trajs = []

        for iter in range(self.config.MAX_ITER):
            # Warps to be used for start of beam updates
            partial_trajs_list = _extract_topk(partial_trajs, self.config.BEAM_SIZE)
            partial_trajs = list()
            warps = []
            for traj in partial_trajs_list:
                warps.append(traj.warps[-1])

            # Do prediction with each candidate warp
            all_candidates, all_scores = [], []
            for warp in warps:
                curr_candidates, curr_scores = [], []
                patch = self._get_region(frame, warp)
                deltaI = np.expand_dims((patch - self.template).reshape(-1), axis=0)

                for learner in self.learners:
                    # Produce p to update current warp
                    p = learner.predict(deltaI).squeeze(0)
                    update_warp = utils.make_hom(p)

                    # Candidate updated warp
                    candidate_warp = np.matmul(warp, update_warp)
                    candidate_warp = utils.normalize_hom(candidate_warp)

                    # Get candidate patch
                    candidate_patch = self._get_region(frame, candidate_warp)

                    # Image similarity score
                    score = utils.SSD(candidate_patch, self.template)
                    curr_scores.append(score)
                    curr_candidates.append(candidate_warp)

                all_candidates.append(curr_candidates)
                all_scores.append(curr_scores)

            # Cumulate beam
            for i, partial_update in enumerate(partial_trajs_list):
                candidates, scores = all_candidates[i], all_scores[i]

                # Topk candidates
                most_likely_updates = np.argsort(scores)[:self.config.BEAM_SIZE]

                for u in most_likely_updates:
                    warp, score = candidates[u], scores[u]

                    warps = partial_update.warps + [warp]
                    score = partial_update.score + score

                    if iter == self.config.MAX_ITER - 1:
                        # End of iteration
                        beam = Trajectory(warps, score)
                        complete_trajs.append(beam)
                    else:
                        # Construct beam traj for next iteration
                        beam = Trajectory(warps, score)
                        partial_trajs.append(beam)

            if len(partial_trajs) == 0:
                # We have run out of partial candidates; happens when beam_size = 1.
                break

        # If we have no complete trajs then fall back to the partial trajs.
        if not len(complete_trajs):
            complete_trajs = partial_trajs

        # Final returns
        complete_trajs = _extract_topk(complete_trajs, self.config.BEAM_SIZE)

        # For tracker update
        # Choose the trajectory with the minimum SSD score as the optimal one
        scores = [u.score for u in complete_trajs]
        idx = np.argmin(scores)
        optimal_traj = complete_trajs[idx]
        self.warp = optimal_traj.warps[-1]

        return optimal_traj, complete_trajs

    def update(self, 
               frame):
        """Produce updated tracked region.
        """
        if not self.initialized:
            raise Exception('Tracker uninitialized!')
        
        if self.config.SEARCH_MODE == 'beam':
            traj, beam_trajs = self._beam_search(frame)
        else:
            traj = self._greedy_search(frame)

        if self.config.DEBUG and traj is not None:
            # Visualize trajectory used for update
            all_corners = [np.round(utils.apply_to_pts(warp, utils._SQUARE)).astype(int) for warp in traj.warps]
            vis = frame.copy()
            for corners in all_corners:
                vis = visualize.draw_region(vis, corners, color=(0, 0, 255))
            plt.imshow(vis)
            plt.title('Updating Trajectory')
            plt.pause(0.001)
            plt.clf()

            # Store trajectory and corners at this stage of updates
            self._trajectories.append(traj)

        return np.round(utils.apply_to_pts(self.warp, utils._SQUARE)).astype(int)