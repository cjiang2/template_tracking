"""
Tracking Python
Code for visualization.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_region(img, corners, color=(0, 255, 0), thickness=2):
    """
    Draw the bounding box specified by the given corners
    of shape (2, 4).
    corners: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] four rectangle 
             corner coordinates.
    """
    if len(img.shape) < 3:
        vis = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    corners = np.round(corners).astype(int) # Force int
    for i in range(4):
        p1 = (corners[0, i], corners[1, i])
        p2 = (corners[0, (i + 1) % 4], corners[1, (i + 1) % 4])
        cv2.line(vis, p1, p2, color, thickness)
    return vis

def visualize_iteration(config, 
                        tracker, 
                        idx, 
                        frame_1):
    """Visualize corners generated during a single tracking
    iteration path planning in one continuous frame update.
    NOTE: debug option must be on! 
    Args:
        config: configuration class.
        tracker: any tracker class.
        idx: frame no index to be visualize.
        frame_1: next 'idx + 1' consecutive frame.
    """
    if config.DEBUG != True:
        raise Exception('Debug option must be on! ')

    # Get corners and trajectories from the specified idx
    temp_corners, trajectories = tracker._all_corners[idx], tracker._trajectories[idx]
    init_corners = tracker._all_corners[idx-1][-1]

    # Visualization
    vis = draw_region(frame_1, init_corners)
    for corners in temp_corners:
        vis = draw_region(vis, corners)
    
    # Save the visualization result
    plt.imshow(vis)
    plt.show()
    cv2.imwrite('vis_{}.jpg'.format(idx), vis)

    return