import cv2, numpy as np

def distance_from_edges(edge_img, mask_size=5):
    """Compute distance transform to nearest edge pixel with improved precision.
    
    Input:
        edge_img: uint8 binary edge map (255=edge, 0=background) or grayscale
        mask_size: mask size for distance transform (3, 5, or cv2.DIST_MASK_PRECISE)
                   Larger values give better accuracy but are slower.
                   Default 5 for better precision (was 3).
    Return:
        dt: float32 distance (same size), where small values lie on/near edges.
    Note:
        cv2.distanceTransform computes distance to zero-pixels.
        We invert so that edges become zeros, then run DT.
    """
    if edge_img.ndim == 3:
        edge_img = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
    # ensure binary: edges->255, background->0
    e = (edge_img > 0).astype(np.uint8) * 255
    inv = (255 - e)
    # Use maskSize=5 for better precision (uses 5x5 mask instead of 3x3)
    # This gives more accurate distance values which should improve chamfer matching
    dt = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=mask_size)
    return dt.astype(np.float32)

def normalize_img(x):
    x = x.astype(np.float32)
    m, M = x.min(), x.max()
    if M - m < 1e-8:
        return np.zeros_like(x, np.float32)
    return (x - m) / (M - m)
