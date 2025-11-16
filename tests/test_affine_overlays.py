import sys
import os

# 確保能找到 sharp 模組
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import cv2
from sharp.affine_match import create_matches_overlay, create_reprojection_overlay

def _dummy_result():
    template = np.zeros((40, 60), dtype=np.uint8)
    scene = np.zeros((50, 70), dtype=np.uint8)

    kp1 = [cv2.KeyPoint(10.0, 12.0, 1), cv2.KeyPoint(30.0, 20.0, 1)]
    kp2 = [cv2.KeyPoint(15.0, 18.0, 1), cv2.KeyPoint(40.0, 25.0, 1)]

    matches = [
        cv2.DMatch(_queryIdx=0, _trainIdx=0, _imgIdx=0, _distance=0.1),
        cv2.DMatch(_queryIdx=1, _trainIdx=1, _imgIdx=0, _distance=0.2),
    ]

    affine_matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 4.0]], dtype=np.float32)

    result = {
        'affine_matrix': affine_matrix,
        'keypoints_template': kp1,
        'keypoints_image': kp2,
        'matches': matches,
        'inliers_mask': np.array([True, False], dtype=bool),
        'num_inliers': 1,
        'num_matches': 2,
        'params': {
            'tx': 5.0,
            'ty': 4.0,
            'theta_deg': 0.0,
            'theta_rad': 0.0,
            'scale': 1.0,
            'scale_x': 1.0,
            'scale_y': 1.0,
        }
    }

    return template, scene, result


def test_matches_overlay_has_alpha_content():
    template, scene, result = _dummy_result()
    overlay = create_matches_overlay(template, scene, result, show_all=True)

    assert overlay.dtype == np.uint8
    expected_h = max(template.shape[0], scene.shape[0])
    expected_w = template.shape[1] + scene.shape[1]
    assert overlay.shape == (expected_h, expected_w, 4)
    assert overlay[..., 3].max() == 255


def test_reprojection_overlay_dimensions():
    template, scene, result = _dummy_result()
    overlay = create_reprojection_overlay(template, scene, result)

    assert overlay.shape == (scene.shape[0], scene.shape[1], 4)
    assert overlay[..., 3].max() == 255
