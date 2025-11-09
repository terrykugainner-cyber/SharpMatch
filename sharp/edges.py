import cv2
import numpy as np

def canny_edges(img, low_thresh=50, high_thresh=120, aperture=3, L2gradient=True):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(img, low_thresh, high_thresh, apertureSize=aperture, L2gradient=L2gradient)
