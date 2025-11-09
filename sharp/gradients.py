import cv2
import numpy as np

def gradient_mag_dir(img, ksize=3, use_scharr=False):
    """Compute gradient magnitude and direction (radians, -pi..pi)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if use_scharr:
        gx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    else:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=False)
    ang = ang - np.pi  # map to (-pi, pi]
    return mag, ang
