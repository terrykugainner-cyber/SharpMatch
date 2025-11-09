import numpy as np, cv2
from sharp.distance import distance_from_edges

def test_distance_nonneg():
    img = np.zeros((64,64), np.uint8)
    cv2.line(img, (10,10), (50,50), 255, 1)
    dt = distance_from_edges(img)
    assert dt.min() >= 0
    assert dt.shape == img.shape
