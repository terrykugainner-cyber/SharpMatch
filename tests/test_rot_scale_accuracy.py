import numpy as np, cv2
from sharp.shape_model import create
from sharp.shape_match import chamfer_coarse, topk_peaks
from sharp.refine import precompute_fields, grid_search_theta_scale, refine_pose_gauss_newton

def test_rot_scale_recovery_small():
    img = np.full((200,240), 30, np.uint8)
    cv2.rectangle(img, (90,70), (150,130), 220, -1)  # template region
    roi = (90,70,60,60)
    model = create(img, roi, edge_threshold=(5,20), sampling_step=1, use_polarity=False)
    H,W = img.shape[:2]
    cx, cy = W//2, H//2
    theta_gt = np.deg2rad(45.0); s_gt = 1.2
    A = np.array([[s_gt*np.cos(theta_gt), -s_gt*np.sin(theta_gt), cx],
                  [s_gt*np.sin(theta_gt),  s_gt*np.cos(theta_gt),  cy]], np.float32)
    scene = cv2.warpAffine(img, A, (W,H), flags=cv2.INTER_LINEAR)
    scoremap, (ox,oy) = chamfer_coarse(model, scene, roi=(0,0,W,H))
    tx,ty,_ = topk_peaks(scoremap, K=1)[0]
    # Chamfer 匹配給出的是模板中心位置
    tpl_h, tpl_w = model['template'].shape[:2]
    template_center_x = tpl_w // 2 + tx + ox
    template_center_y = tpl_h // 2 + ty + oy
    fields = precompute_fields(scene, edge_threshold=(5,20))
    angs = np.deg2rad(np.linspace(-180,180,37)); scales = np.linspace(0.8,1.4,13)
    # 傳入模板中心位置，函數內部會計算旋轉後的模板左上角
    cands = grid_search_theta_scale(model, fields, (template_center_x, template_center_y), angs, scales, topk=3)
    sc,x,y,th,s = cands[0]
    x1,y1,th1,s1,sc1 = refine_pose_gauss_newton(model, fields, (x,y,th,s), iters=15, sample_max=500)
    err_deg = abs((np.rad2deg(th1) - 45.0 + 180) % 360 - 180)
    assert err_deg < 10.0
    assert abs(s1 - 1.2) < 0.2
