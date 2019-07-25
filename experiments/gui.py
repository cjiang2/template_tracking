"""
OpenCV GUI Wrapper for real-time tracking.

"""

import os
import sys

import cv2
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import tracking utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from tracking.hyperplane import HyperplaneTracker
from tracking.nearest_neighbor import NNTracker
from tracking.visualize import draw_region
from tracking.config import Config

class GUIWrapper:
    def __init__(self,
                 tracker,
                 window_shape=(640, 480),
                 window_name='Tracking Window'):
        self.tracker = tracker
        self.window_shape = window_shape
        self.window_name = window_name

        self.img = None
        self.img_gray = None
        self.initialized = False
        self.clicks = []
        self.corners = None
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_event)

    def init_feed(self, 
                  feed_id=0):
        """Helper func to initialize camera feed through OpenCV.
        """
        print('Initializing camera feed...')
        cap = cv2.VideoCapture(feed_id)
        if not cap.isOpened():
            raise Exception("Could not open video device")

        # Set window size
        if self.window_shape is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_shape[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_shape[1])

        # Set FPS to 30
        cap.set(cv2.CAP_PROP_FPS, 30)

        print('Done.')
        return cap

    def mouse_event(self, event, x, y, flags, param):
        """OpenCV mouse event.
        """
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicks) < 4:
            self.clicks.append([x, y])
            if len(self.clicks) == 4:
                self.corners = np.array(self.clicks).T
                self.initialized = True

    def display(self):
        """Display current image frame.
        """
        vis = self.img.copy()
        if self.initialized:
            vis = draw_region(vis, self.corners, color=(0, 0, 255))
        else:
            for pt in self.clicks:
                cv2.circle(vis, (pt[0], pt[1]), 2, (255, 0, 0), 2)
        cv2.imshow(self.window_name, vis)

    def on_frame(self,
                 img):
        """Process current frame from feed.
        """
        self.img = img
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)   
        self.img_gray = cv2.GaussianBlur(self.img_gray, (5,5), 3)
        if self.initialized:
            if self.tracker.initialized:
                self.corners = self.tracker.update(self.img_gray)
            else:
                self.tracker.initialize(self.img_gray, self.corners)
        self.display()

    def release(self):
        """Clear window.
        """
        cv2.destroyWindow(self.window_name)


if __name__ == '__main__':
    # Test
    #nntracker = NNTracker(debug=True)
    #gui = GUIWrapper(tracker=nntracker)

    hyperplane = HyperplaneTracker(debug=True)
    gui = GUIWrapper(tracker=hyperplane)

    cap = gui.init_feed(1)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        gui.on_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    gui.release()
    
