import numpy as np, cv2
from sharp.shape_model import create

def test_model_points_exist():
    img = np.zeros((100,120), np.uint8)
    cv2.rectangle(img, (40,30), (80,70), 200, -1)
    m = create(img, (40,30,40,40), edge_threshold=(10,40), sampling_step=1)
    assert len(m['points']) > 0
    assert m['template'].shape[0] == 40 and m['template'].shape[1] == 40
