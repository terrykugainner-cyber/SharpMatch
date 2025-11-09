import numpy as np, cv2
from sharp.shape_model import create
from sharp.refine import precompute_fields, grid_search_theta_scale, refine_pose_gauss_newton
def test_refine_improves_or_equal():
    img = np.zeros((160,200), np.uint8)
    cv2.rectangle(img, (80,60), (130,110), 200, -1)
    roi=(80,60,50,50)
    model = create(img, roi, edge_threshold=(5,20), sampling_step=1, use_polarity=False)
    fields = precompute_fields(img, edge_threshold=(5,20))
    # 測試用的固定位置 (x0, y0) 是模板左上角
    # 需要轉換為模板中心位置
    x0, y0 = 78, 62
    tpl_h, tpl_w = model['template'].shape[:2]
    template_center_x = x0 + tpl_w // 2
    template_center_y = y0 + tpl_h // 2
    angs = np.deg2rad(np.linspace(-30,30,7)); scales = np.linspace(0.9,1.1,5)
    # 傳入模板中心位置，函數內部會計算旋轉後的模板左上角
    cands = grid_search_theta_scale(model, fields, (template_center_x, template_center_y), angs, scales, topk=3)
    sc0,x,y,th,s = cands[0]
    x1,y1,th1,s1,sc1 = refine_pose_gauss_newton(model, fields, (x,y,th,s), iters=10, sample_max=300)
    assert sc1 >= sc0 - 1e-4
