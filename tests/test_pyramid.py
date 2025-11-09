import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2, numpy as np
from sharp.pyramid import build_pyramid

def test_build_pyramid_shapes():
    img = np.zeros((200,300), np.uint8)
    pyr = build_pyramid(img, num_levels=4, scale=0.5)
    assert len(pyr) == 4
    h0, w0 = pyr[0].shape
    for i in range(1,4):
        hi, wi = pyr[i].shape
        assert hi <= h0 and wi <= w0
