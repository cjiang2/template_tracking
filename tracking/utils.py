import os
import glob
from typing import Tuple
import time

import numpy as np
import cv2

# -----
# Inverse Compositional Updates

# Use a unit square to parameterize warps
# _SQUARE = np.float32([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]).T      # Centered
_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]).T    # Uncentered

def square_to_corner_warp(
    corner: np.ndarray,
    ) -> np.ndarray:
    """Computes the homography from the unit square to
    the quadrilateral corner (2, 4).
    """
    warp, _ = cv2.findHomography(_SQUARE.T, np.float32(corner.T))
    return warp

def generate_random_warp(
    sigma_d: float, 
    sigma_t: float,
    ) -> np.ndarray:
    """Disturb the unit square and get homography.
    """
    disturbed = np.random.normal(0, sigma_d, (2, 4)) + np.random.normal(0, sigma_t, (2, 1)) + _SQUARE
    warp, _ = cv2.findHomography(_SQUARE.T, disturbed.T)
    return warp

def corner_to_square(
    warp: np.ndarray
    ) -> np.ndarray:
    return apply_homography(warp, _SQUARE)

def apply_homography(
    warp: np.ndarray, 
    pts: np.ndarray,
    ):
    """Apply a geometric transformation on point set of shape (2, n).
    """
    (h, w) = pts.shape    
    pts = homogenize(pts)
    result = np.matmul(warp, pts)
    result[:h] = result[:h] / result[-1]
    result[np.isnan(result)] = 0
    return result[:h]


# -----
# Templates

def get_birdeye_view_(
    img: np.ndarray,
    warp: np.ndarray,
    output_resolution: Tuple[int],
    interpolation: int = cv2.INTER_LINEAR,
    ):
    """Sample a bird-eye view of the region, 
    specified by a (2, 4) corner.
    NOTE: Work with image coordinates to sample region. Slower.
    """
    start = time.process_time()
    # Destinating shape
    height, width = output_resolution[0], output_resolution[1]

    # Construct a [0, 1] meshgrid
    Xq = np.linspace(0, 1, width, dtype=np.float32)
    Yq = np.linspace(0, 1, height, dtype=np.float32)
    Xq, Yq = np.meshgrid(Xq, Yq)

    # Square to corner warp / homography
    x = np.stack([Xq.reshape(-1), Yq.reshape(-1)], axis=1)[:, np.newaxis, :]
    x = cv2.perspectiveTransform(x, warp)
    Xq, Yq = x[:, 0, 0].reshape((height, width)), x[:, 0, 1].reshape((height, width))

    # Sample region
    region = cv2.remap(img, Xq, Yq, interpolation=interpolation)
    # print(time.process_time() - start)

    return region


def get_birdeye_view(
    img: np.ndarray, 
    corner: np.ndarray, 
    output_resolution: Tuple[int],
    interpolation=cv2.INTER_LINEAR,
    ):
    """Sample a bird-eye view of the region, 
    specified by a (2, 4) corner.
    """
    start = time.process_time()
    height, width = output_resolution[0], output_resolution[1]

    # Homography for image warping
    # Origin coordinate
    src = np.float32(corner.T)
    dst = np.array([[0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]], dtype=np.float32)
    dst[dst < 0] = 0

    # Warp from img coords to bird-eye view coords
    M, _ = cv2.findHomography(src, dst)
    region = cv2.warpPerspective(img, M, (width, height), 
                borderMode=cv2.BORDER_CONSTANT, borderValue=0, flags=interpolation)
    # print(time.process_time() - start)

    return region


# -----
# Edge templates

def get_bounding_quadrilateral(
    mask: np.ndarray,
    ):
    cnt = cv2.findNonZero(mask)
    # [x, y, w, h] = cv2.boundingRect(cnt)
    # corner = np.int32([
    #     [x-10, x + w + 10, x + w + 10, x - 10],
    #     [y - 10, y - 10, y + h + 10, y + h + 10],
    # ])
    corner = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(int).T
    return corner



# -----
# Image Coordinates

def homogenize(corner: np.ndarray):
    """Transform points (n, m) into their homogeneous coordinate 
    form of (n+1, m).
    """
    (h, w) = corner.shape
    results = np.empty((h+1, w))
    results[:h] = corner
    results[-1].fill(1)
    return results

def dehomogenize(corner: np.ndarray):
    """Transform and denormalize points (n+1, m) into their cartesian 
    coordinate form (n, m).
    """
    (h, w) = corner.shape
    results = np.empty((h-1,w))
    results[:h-1] = corner[:h-1] / corner[h-1]
    return results

def normalize_hom(hom):
    """Normalize a homography matrix.
    """
    return hom / hom[2, 2]


# -----
# Data loading operations

def read_annotation(filename: str):
    """Read ground truth annotation data from MTF.
    Dataset available at:
        https://webdocs.cs.ualberta.ca/~vis/mtf/
    """
    if not os.path.isfile(filename):
        print("Tracking data file not found:\n ", filename)
        sys.exit()

    data_file = open(filename, 'r')
    data_array = []
    line_id = 0
    for line in data_file:
        if line_id > 0:
            line = line.strip().split()
            coordinate = np.array([[float(line[1]), float(line[3]), float(line[5]), float(line[7])],
                                [float(line[2]), float(line[4]), float(line[6]), float(line[8])]])
            coordinate = np.round(coordinate)
            data_array.append(coordinate.astype(int))
        line_id += 1
    data_file.close()
    return np.array(data_array)

def load_quadrilateral_video(
    path: str,
    filename: str = "frame%05d.jpg"
    ):
    src_fname = os.path.join(path, filename)

    annot_filename = path[:-1] + path[-1] if path[-1] != os.sep else path[:-1]
    annot_filename += ".txt"
    gt = read_annotation(annot_filename)

    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        raise Exception('The video file ', src_fname, ' could not be opened')
    
    return cap, gt

def load_edge_video(
    path: str,
    filename: str = "%04d.jpg",
    ):
    # n_imgs = len(glob.glob(os.path.join(path, "imgs", "*" + filename[-4:])))
    src_fname = os.path.join(path, "imgs", filename)
    cap = cv2.VideoCapture()
    if not cap.open(src_fname):
        raise Exception('The video file ', src_fname, ' could not be opened')
    
    return cap
