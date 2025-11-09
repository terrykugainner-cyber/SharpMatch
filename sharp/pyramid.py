import cv2
import numpy as np

def build_pyramid(img, num_levels=4, scale=0.5, blur=True):
    """Build image pyramid (grayscale).

    Args:
        img: np.uint8 or float32 grayscale image
        num_levels: number of levels including base
        scale: downscale factor per level (0<scale<1)
        blur: optional pre-blur before downscale to reduce aliasing
    Returns:
        list of images [level0, level1, ...]
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pyr = [img.copy()]
    cur = img.copy()
    for _ in range(1, num_levels):
        if blur:
            cur = cv2.GaussianBlur(cur, (5,5), 0)
        new_size = (max(1, int(cur.shape[1]*scale)), max(1, int(cur.shape[0]*scale)))
        cur = cv2.resize(cur, new_size, interpolation=cv2.INTER_AREA)
        pyr.append(cur)
    return pyr
