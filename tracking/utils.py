"""
Tracking Python
Utils code enabling various functionalities.
"""

import numpy as np
import cv2

# ------------------------------------------------------------
# Functions for sampling, homogeneous coordinates
# ------------------------------------------------------------

def sample_region(img, 
                  corners, 
                  region_shape=None,
                  method=0, 
                  interpolation=cv2.INTER_NEAREST,
                  Np=None):
    """Sample corners region as Bird-eye view patch. Left top point of the 
    rect corners is defined as the origin coordinate (0, 0).
    Args:
        img: image to be sampled
        corners: (2, 4) coordinates.
        region_shape: (height, width) of the final sampled region.
        method: 0, cv2.RANSAC or other options here.
        interpolation: interpolation for perspective transformation.
    Returns:
        region: sampled bird-eye view img region.
    """
    if region_shape is None:
        # Top-left, Top-right, Bottom-right, Bottom-left coordinates
        tl, tr, br, bl = corners.T
        # Calculate widths and heights of the corners rect
        width_1 = np.sqrt(np.square(tl[0] - tr[0]) + np.square(tl[1] - tr[1]))
        width_2 = np.sqrt(np.square(bl[0] - br[0]) + np.square(bl[1] - br[1]))
        height_1 = np.sqrt(np.square(tl[0] - bl[0]) + np.square(tl[1] - bl[1]))
        height_2 = np.sqrt(np.square(tr[0] - br[0]) + np.square(tr[1] - br[1]))

        # Choose the maximum width and height
        width = int(np.round(max(width_1, width_2)))
        height = int(np.round(max(height_1, height_2)))
    else:
        height, width = region_shape

    # Homography for image warping
    # Origin coordinate
    src = np.float32(corners.T)
    dst = np.array([[0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]], dtype=np.float32)
    dst[dst < 0] = 0

    # Birds-eye-view for the patch image based on corners
    M, _ = cv2.findHomography(src, dst, method)   # OpenCV accepts (4, 2) only
    region = cv2.warpPerspective(img, 
                                 M, 
                                 (width, height), 
                                 flags=interpolation, 
                                 borderMode=cv2.BORDER_REPLICATE)
                                 #borderMode=cv2.BORDER_CONSTANT,
                                 #borderValue=0)
    
    # Perform grid sampling if specified
    if Np is not None:
        assert width == height  # Ensure it's a rectangle patch for now
        # Generate mesh grid
        grid_size = int(np.round(np.sqrt(Np)))
        x = np.linspace(0, width - 1, num=grid_size, dtype=int)
        y = np.linspace(0, height - 1, num=grid_size, dtype=int)
        xv, yv = np.meshgrid(x, y, sparse=False)
        # Subsample region
        region = region[yv, xv]

    return region

def polys_to_mask(img,
                  polys):
    """Convert list shape (n, 2) of polygon points to a binary mask.
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polys], 255)
    return mask

def homogenize(corners):
    """Transform points (n, m) into their homogeneous coordinate 
    form of (n+1, m).
    """
    (h, w) = corners.shape
    results = np.empty((h+1, w))
    results[:h] = corners
    results[-1].fill(1)
    return results

def dehomogenize(corners):
    """Transform and denormalize points (n+1, m) into their cartesian 
    coordinate form (n, m).
    """
    (h, w) = corners.shape
    results = np.empty((h-1,w))
    results[:h-1] = corners[:h-1] / corners[h-1]
    return results

# ------------------------------------------------------------
# Functions for normalized homography motion generation and 
# Inverse Compositional updates
# From Martin's NN paper.
# ------------------------------------------------------------

_SQUARE = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T
_SQUARE_AFF = np.array([[-.5,-.5],[.5,-.5],[-.5,.5]]).T

def compute_homography(in_pts, 
                       out_pts):
    """Normalized Direct Linear Transformation to compute homography.
    """
    num_pts = in_pts.shape[1]
    constraint_matrix = np.empty((num_pts*2, 9), dtype=np.float64)
    for i in range(num_pts):
        r1 = 2*i
        constraint_matrix[r1,0] = 0
        constraint_matrix[r1,1] = 0
        constraint_matrix[r1,2] = 0
        constraint_matrix[r1,3] = -in_pts[0,i]
        constraint_matrix[r1,4] = -in_pts[1,i]
        constraint_matrix[r1,5] = -1
        constraint_matrix[r1,6] = out_pts[1,i] * in_pts[0,i]
        constraint_matrix[r1,7] = out_pts[1,i] * in_pts[1,i]
        constraint_matrix[r1,8] = out_pts[1,i]

        r2 = 2*i + 1
        constraint_matrix[r2,0] = in_pts[0,i]
        constraint_matrix[r2,1] = in_pts[1,i]
        constraint_matrix[r2,2] = 1
        constraint_matrix[r2,3] = 0
        constraint_matrix[r2,4] = 0
        constraint_matrix[r2,5] = 0
        constraint_matrix[r2,6] = -out_pts[0,i] * in_pts[0,i]
        constraint_matrix[r2,7] = -out_pts[0,i] * in_pts[1,i]
        constraint_matrix[r2,8] = -out_pts[0,i]
    U,S,V = np.linalg.svd(constraint_matrix)
    H = V[8].reshape(3,3) / V[8][8]
    return H

def apply_homography(homography, 
                     pts):
    """Apply homography on point set of shape (2, n).
    """
    (h, w) = pts.shape    
    pts = homogenize(pts)
    result = np.matmul(homography, pts)
    result[:h] = result[:h] / result[h]
    result[np.isnan(result)] = 0
    return result[:h]

def normalize_hom(homography):
    """Normalize homography matrix.
    """
    return homography / homography[2, 2]

def square_to_corners_warp(corners):
    """Computes the homography from the centered unit square to
       the quadrilateral defined by the corners.
    """
    return compute_homography(_SQUARE, corners)

def random_homography(sigma_t, 
                      sigma_d):
    """Generate a random homography motion.
    """
    disturbed = np.random.normal(0, sigma_d, (2, 4)) + np.random.normal(0, sigma_t, (2, 1)) + _SQUARE
    H = compute_homography(_SQUARE, disturbed)
    return H

def make_hom(p):
    """Convert regression result back into a homography.
    For normalized homography sampling only.
    """
    hom = np.identity(3).astype(np.float32)
    hom[0,0] = p[0];hom[0,1] = p[1];hom[0,2] = p[2]
    hom[1,0] = p[3];hom[1,1] = p[4];hom[1,2] = p[5]
    hom[2,0] = p[6];hom[2,1] = p[7]
    return hom

# ------------------------------------------------------------
# Functions for unnormalized uniform homography motion generation
# Described in the original hyperplane tracking paper.
# ------------------------------------------------------------

def random_uniform_homography(dist_range, 
                              corners,
                              inv_warp=True,
                              method=0):
    """Generate a random uniform homography motion.
    """
    assert dist_range >= 0
    delta_p = np.random.uniform(low=-dist_range, 
                                high=dist_range, 
                                size=(2, 4))
    disturbed = corners + delta_p
    if inv_warp:
        M, _ = cv2.findHomography(disturbed.T, corners.T, method)
    else:
        M, _ = cv2.findHomography(corners.T, disturbed.T, method)
    return M, delta_p

# ------------------------------------------------------------
# Functions for image operation.
# ------------------------------------------------------------

def normalize_zscore(intensity):
    """zero mean, unit variance normalization.
    """
    intensity = (intensity - np.mean(intensity)) / np.std(intensity, ddof=1)
    return intensity

def normalize_minmax(intensity):
    """Min-max normalization.
    """
    intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    return intensity
