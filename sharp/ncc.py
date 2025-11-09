import cv2
import numpy as np
from .pyramid import build_pyramid
import multiprocessing as mp
from functools import partial

def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def ncc_scoremap(tpl, img, roi=None, mask=None):
    """Return normalized cross-correlation score map in [0,1].
    
    Implements ZNCC (Zero-mean Normalized Cross-Correlation) with optional
    ROI and mask support. When mask is provided, only non-zero pixels
    participate in computing mean, variance, and correlation (ZNCC with mask).
    
    Args:
        tpl: template image (gray/BGR)
        img: search image (gray/BGR)
        roi: optional (x,y,w,h) tuple to restrict search region in img
        mask: optional mask for template, same size as tpl (0=ignore, non-zero=use)
              When mask is used, only mask=1 pixels participate in ZNCC calculation
    
    Returns:
        scoremap: float32 array in [0,1], larger is better
                  If roi is provided, scoremap size matches (img_h-tpl_h+1, img_w-tpl_w+1)
                  where img_h, img_w are the roi dimensions
    """
    tpl = _to_gray(tpl)
    img = _to_gray(img)
    
    # Apply ROI if specified
    if roi is not None:
        x, y, w, h = roi
        img_crop = img[y:y+h, x:x+w]
    else:
        img_crop = img
    
    # Prepare mask if provided
    if mask is not None:
        # Ensure mask is uint8 and same size as template
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        if mask.shape != tpl.shape:
            raise ValueError(f"mask shape {mask.shape} must match template shape {tpl.shape}")
        # OpenCV matchTemplate with mask: only non-zero mask pixels participate in ZNCC
        res = cv2.matchTemplate(img_crop, tpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    else:
        res = cv2.matchTemplate(img_crop, tpl, cv2.TM_CCOEFF_NORMED)
    
    # map from [-1,1] to [0,1]
    res = (res + 1.0) * 0.5
    return res.astype(np.float32)

def pyramid_ncc_search(tpl, img, roi=None, mask=None, num_levels=4, scale=0.5):
    """Coarse-to-fine NCC search with ROI and mask support. Returns (x,y,score).
    
    Implements ZNCC (Zero-mean Normalized Cross-Correlation) with optional
    ROI and mask support. When mask is provided, only non-zero pixels
    participate in computing mean, variance, and correlation (ZNCC with mask).
    
    Args:
        tpl: template image (gray/BGR)
        img: search image (gray/BGR)
        roi: optional (x,y,w,h) in base image, restrict search region
        mask: optional mask for template, same size as tpl (0=ignore, non-zero=use)
              When mask is used, only mask=1 pixels participate in ZNCC calculation
        num_levels: number of pyramid levels
        scale: pyramid scale factor (typically 0.5)
    
    Returns:
        (x, y, score): best match location (in original image coordinates) and score [0,1]
                      x, y are adjusted by ROI offset (offx, offy)
    """
    tpl = _to_gray(tpl)
    img = _to_gray(img)
    
    # Prepare mask if provided
    if mask is not None:
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        if mask.shape != tpl.shape:
            raise ValueError(f"mask shape {mask.shape} must match template shape {tpl.shape}")
        # Build mask pyramid (automatically scaled for each level)
        mask_pyr = build_pyramid(mask, num_levels=num_levels, scale=scale)
    else:
        mask_pyr = [None] * num_levels
    
    # Apply ROI if specified (crop image and track offset)
    if roi is not None:
        x,y,w,h = roi
        img_crop = img[y:y+h, x:x+w]
        offx, offy = x, y
    else:
        img_crop = img
        offx = offy = 0

    tpl_pyr = build_pyramid(tpl, num_levels=num_levels, scale=scale)
    img_pyr = build_pyramid(img_crop, num_levels=num_levels, scale=scale)

    # start from top (smallest) level
    sx = sy = 0
    for lvl in range(num_levels-1, -1, -1):
        tpl_l = tpl_pyr[lvl]
        img_l = img_pyr[lvl]
        mask_l = mask_pyr[lvl]
        
        # Scale up position from coarser level
        if lvl != num_levels-1:
            sx *= 2.0  # since scale=0.5, 1/scale = 2
            sy *= 2.0
        
        # Define search window
        if lvl == num_levels-1:
            # Top level: search entire image
            search = img_l
            ox = oy = 0
        else:
            # Refine: search around previous result
            H, W = img_l.shape[:2]
            th, tw = tpl_l.shape[:2]
            cx = int(sx)
            cy = int(sy)
            r = max(th, tw)  # Search radius
            x0 = max(0, cx - r)
            y0 = max(0, cy - r)
            x1 = min(W, cx + r + tw)
            y1 = min(H, cy + r + th)
            search = img_l[y0:y1, x0:x1]
            ox, oy = x0, y0

        # Template matching at this level with mask support
        # OpenCV matchTemplate with mask: only non-zero mask pixels participate in ZNCC
        if mask_l is not None:
            score = cv2.matchTemplate(search, tpl_l, cv2.TM_CCOEFF_NORMED, mask=mask_l)
        else:
            score = cv2.matchTemplate(search, tpl_l, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(score)
        sx = maxLoc[0] + ox
        sy = maxLoc[1] + oy

    # back to original coordinates (add ROI offset)
    x = int(round(sx)) + offx
    y = int(round(sy)) + offy
    final_score = (maxVal + 1.0) * 0.5  # map [-1,1]→[0,1]
    return x, y, float(final_score)


def ncc_score_rotated(tpl, img, center_x, center_y, theta, scale, mask=None):
    """計算旋轉和縮放後的模板在指定位置的 NCC 分數。
    
    Args:
        tpl: 模板圖像 (gray/BGR)
        img: 場景圖像 (gray/BGR)
        center_x, center_y: 模板中心在場景中的位置
        theta: 旋轉角度（弧度）
        scale: 縮放因子
        mask: 可選的模板 mask
    
    Returns:
        score: NCC 分數 [0, 1]，越大越好
    """
    tpl = _to_gray(tpl)
    img = _to_gray(img)
    
    h, w = img.shape[:2]
    tpl_h, tpl_w = tpl.shape[:2]
    
    # 計算旋轉和縮放後的模板尺寸
    new_w = int(tpl_w * scale)
    new_h = int(tpl_h * scale)
    
    # 如果尺寸太小，返回低分
    if new_w < 1 or new_h < 1:
        return 0.0
    
    # 計算旋轉矩陣（以模板中心為原點）
    #M = cv2.getRotationMatrix2D((tpl_w / 2, tpl_h / 2), np.rad2deg(theta), scale)

    # 反轉角度符號，使其與用戶視角的旋轉方向一致
    # 正角度 = 逆時針旋轉（從用戶視角）
    M = cv2.getRotationMatrix2D((tpl_w / 2, tpl_h / 2), -np.rad2deg(theta), scale)
    
    # 旋轉和縮放模板
    #tpl_rotated = cv2.warpAffine(tpl, M, (new_w, new_h), 
    #                             flags=cv2.INTER_LINEAR, 
    #                             borderMode=cv2.BORDER_CONSTANT)

    # 在 ncc_score_rotated 函數中，將 INTER_LINEAR 改為 INTER_CUBIC
    tpl_rotated = cv2.warpAffine(tpl, M, (new_w, new_h), 
                                 flags=cv2.INTER_CUBIC,  # 從 INTER_LINEAR 改為 INTER_CUBIC
                                 borderMode=cv2.BORDER_CONSTANT)
    
    # 旋轉 mask（如果提供）
    mask_rotated = None
    if mask is not None:
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        if mask.shape != tpl.shape:
            raise ValueError(f"mask shape {mask.shape} must match template shape {tpl.shape}")
        mask_rotated = cv2.warpAffine(mask, M, (new_w, new_h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT)
    
    # 計算模板左上角位置
    x0 = int(round(center_x - new_w / 2))
    y0 = int(round(center_y - new_h / 2))
    
    # 檢查邊界
    if x0 < 0 or y0 < 0 or x0 + new_w > w or y0 + new_h > h:
        # 如果超出邊界，需要裁剪
        x1 = max(0, x0)
        y1 = max(0, y0)
        x2 = min(w, x0 + new_w)
        y2 = min(h, y0 + new_h)
        
        # 裁剪模板和 mask
        tpl_crop = tpl_rotated[y1-y0:y2-y0, x1-x0:x2-x0]
        if mask_rotated is not None:
            mask_crop = mask_rotated[y1-y0:y2-y0, x1-x0:x2-x0]
        else:
            mask_crop = None
        
        # 裁剪場景
        img_crop = img[y1:y2, x1:x2]
        
        # 檢查尺寸是否匹配
        if tpl_crop.shape[0] != img_crop.shape[0] or tpl_crop.shape[1] != img_crop.shape[1]:
            return 0.0
        
        # 計算 NCC
        if mask_crop is not None:
            res = cv2.matchTemplate(img_crop, tpl_crop, cv2.TM_CCOEFF_NORMED, mask=mask_crop)
        else:
            res = cv2.matchTemplate(img_crop, tpl_crop, cv2.TM_CCOEFF_NORMED)
        
        score = (res[0, 0] + 1.0) * 0.5
        return float(np.clip(score, 0.0, 1.0))
    
    # 提取場景區域
    img_crop = img[y0:y0+new_h, x0:x0+new_w]
    
    # 計算 NCC
    if mask_rotated is not None:
        res = cv2.matchTemplate(img_crop, tpl_rotated, cv2.TM_CCOEFF_NORMED, mask=mask_rotated)
    else:
        res = cv2.matchTemplate(img_crop, tpl_rotated, cv2.TM_CCOEFF_NORMED)
    
    score = (res[0, 0] + 1.0) * 0.5
    return float(np.clip(score, 0.0, 1.0))


def refine_match_local(tpl, img, center_x, center_y, theta, scale, 
                       ang_range=0.5, scale_range=0.02, pos_range=2.0,
                       ang_steps=11, scale_steps=5, pos_steps=5, mask=None):
    """在最佳候選附近進行局部精細優化。
    
    這是一個三維搜索：位置 (x, y)、角度 (theta)、縮放 (scale)
    用於在找到初步匹配後，進一步提高定位精度。
    
    Args:
        tpl: 模板圖像
        img: 場景圖像
        center_x, center_y: 初始位置
        theta: 初始角度（弧度）
        scale: 初始縮放
        ang_range: 角度搜索範圍（弧度），例如 0.5° = 0.0087 弧度
        scale_range: 縮放搜索範圍，例如 ±0.02
        pos_range: 位置搜索範圍（像素），例如 ±2 像素
        ang_steps: 角度搜索步數
        scale_steps: 縮放搜索步數
        pos_steps: 位置搜索步數（每個方向）
        mask: 可選的模板 mask
    
    Returns:
        (best_score, best_x, best_y, best_theta, best_scale)
    """
    tpl = _to_gray(tpl)
    img = _to_gray(img)
    
    # 將角度範圍從度數轉換為弧度（如果輸入的是度數）
    if ang_range > 0.1:  # 如果看起來像度數，轉換為弧度
        ang_range = np.deg2rad(ang_range)
    
    best_score = 0.0
    best_x, best_y, best_theta, best_scale = center_x, center_y, theta, scale
    
    # 創建搜索網格
    ang_grid = np.linspace(theta - ang_range, theta + ang_range, ang_steps)
    scale_grid = np.linspace(scale - scale_range, scale + scale_range, scale_steps)
    pos_x_grid = np.linspace(center_x - pos_range, center_x + pos_range, pos_steps)
    pos_y_grid = np.linspace(center_y - pos_range, center_y + pos_range, pos_steps)
    
    # 三維搜索（位置、角度、縮放）
    for px in pos_x_grid:
        for py in pos_y_grid:
            for th in ang_grid:
                for s in scale_grid:
                    score = ncc_score_rotated(tpl, img, px, py, th, s, mask)
                    if score > best_score:
                        best_score = score
                        best_x, best_y, best_theta, best_scale = px, py, th, s
    
    return (best_score, best_x, best_y, best_theta, best_scale)


def refine_match_iterative(tpl, img, center_x, center_y, theta, scale,
                           max_iters=5, tol=1e-3, mask=None,
                           initial_ang_range=0.5, initial_scale_range=0.02, initial_pos_range=2.0):
    """使用迭代優化精細匹配（類似於 Gauss-Newton）。
    
    每次迭代都在當前最佳參數附近搜索，搜索範圍隨迭代次數遞減，
    直到收斂或達到最大迭代次數。
    
    Args:
        tpl: 模板圖像
        img: 場景圖像
        center_x, center_y: 初始位置
        theta: 初始角度（弧度）
        scale: 初始縮放
        max_iters: 最大迭代次數
        tol: 收斂容差（分數變化小於此值則停止）
        mask: 可選的模板 mask
        initial_ang_range: 初始角度搜索範圍（度數）
        initial_scale_range: 初始縮放搜索範圍
        initial_pos_range: 初始位置搜索範圍（像素）
    
    Returns:
        (best_score, best_x, best_y, best_theta, best_scale)
    """
    tpl = _to_gray(tpl)
    img = _to_gray(img)
    
    # 將角度範圍從度數轉換為弧度
    if initial_ang_range > 0.1:
        initial_ang_range = np.deg2rad(initial_ang_range)
    
    x, y, th, s = center_x, center_y, theta, scale
    best_score = ncc_score_rotated(tpl, img, x, y, th, s, mask)
    prev_score = best_score
    
    for iteration in range(max_iters):
        # 計算當前迭代的搜索範圍（隨迭代次數遞減）
        decay_factor = 1.0 / (iteration + 1)  # 從 1.0 遞減到 1/max_iters
        ang_range = initial_ang_range * decay_factor
        scale_range = initial_scale_range * decay_factor
        pos_range = initial_pos_range * decay_factor
        
        # 創建搜索網格（步數隨迭代次數遞增，提高精度）
        ang_steps = 5 + iteration * 2  # 從 5 步增加到更多步
        scale_steps = 3 + iteration  # 從 3 步增加
        pos_steps = 3 + iteration  # 從 3 步增加
        
        ang_grid = np.linspace(th - ang_range, th + ang_range, ang_steps)
        scale_grid = np.linspace(s - scale_range, s + scale_range, scale_steps)
        x_grid = np.linspace(x - pos_range, x + pos_range, pos_steps)
        y_grid = np.linspace(y - pos_range, y + pos_range, pos_steps)
        
        # 搜索最佳參數
        found_better = False
        for px in x_grid:
            for py in y_grid:
                for th_new in ang_grid:
                    for s_new in scale_grid:
                        score = ncc_score_rotated(tpl, img, px, py, th_new, s_new, mask)
                        if score > best_score:
                            best_score = score
                            x, y, th, s = px, py, th_new, s_new
                            found_better = True
        
        # 檢查收斂
        score_change = abs(best_score - prev_score)
        if score_change < tol:
            # 分數變化很小，認為已收斂
            break
        
        prev_score = best_score
        
        # 如果沒有找到更好的解，提前退出
        if not found_better:
            break
    
    return (best_score, x, y, th, s)


def _compute_ncc_score_parallel(args):
    """並行計算 NCC 分數的輔助函數
    
    Args:
        args: (tpl, img, center_x, center_y, theta, scale, mask) 元組
    
    Returns:
        (score, center_x, center_y, theta, scale) 元組
    """
    tpl, img, center_x, center_y, theta, scale, mask = args
    score = ncc_score_rotated(tpl, img, center_x, center_y, theta, scale, mask)
    return (score, center_x, center_y, theta, scale)


def grid_search_theta_scale_ncc(tpl, img, center_x, center_y, 
                                  ang_grid, scales, topk=5, mask=None, roi=None,
                                  coarse_first=True, coarse_ratio=0.3,
                                  use_parallel=False, num_workers=None):
    """在角度和縮放空間進行網格搜索，使用 NCC 評分。
    
    支持兩階段搜索：先進行粗搜索快速篩選，再在最佳候選附近進行精細搜索。
    支持並行化計算：使用多進程加速 NCC 計算。
    
    Args:
        tpl: 模板圖像
        img: 場景圖像
        center_x, center_y: 初始模板中心位置
        ang_grid: 角度網格（弧度）
        scales: 縮放因子列表
        topk: 返回前 K 個最佳候選
        mask: 可選的模板 mask
        roi: 可選的搜索區域限制 (x, y, w, h)
        coarse_first: 是否先進行粗搜索（當搜索空間較大時）
        coarse_ratio: 粗搜索保留的候選比例（例如 0.3 表示保留前 30% 的候選）
        use_parallel: 是否使用並行化計算（默認 False）
        num_workers: 並行化時的工作進程數（None 表示自動檢測 CPU 核心數）
    
    Returns:
        List of (score, x, y, theta, scale) tuples, sorted by score descending
    """
    tpl = _to_gray(tpl)
    img = _to_gray(img)
    
    # 應用 ROI
    if roi is not None:
        rx, ry, rw, rh = roi
        img = img[ry:ry+rh, rx:rx+rw]
        center_x -= rx
        center_y -= ry
    
    total_search = len(ang_grid) * len(scales)
    
    # 自動判斷是否使用並行化（任務足夠大時才值得）
    if use_parallel and num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # 保留一個核心給主進程
    
    # 判斷是否需要兩階段搜索
    use_two_stage = (coarse_first and total_search > 50 and 
                     len(ang_grid) > 10 and len(scales) > 3)
    
    # 判斷是否使用並行化（任務足夠大時才值得）
    should_parallel = (use_parallel and total_search > 20 and num_workers > 1)
    
    def compute_candidates_parallel(angle_list, scale_list):
        """並行或串行計算候選列表"""
        if not should_parallel or len(angle_list) * len(scale_list) < 20:
            # 串行計算（小任務或未啟用並行化）
            candidates = []
            for theta in angle_list:
                for scale in scale_list:
                    score = ncc_score_rotated(tpl, img, center_x, center_y, theta, scale, mask)
                    candidates.append((score, center_x, center_y, theta, scale))
            return candidates
        else:
            # 並行計算
            args_list = [(tpl, img, center_x, center_y, theta, scale, mask)
                        for theta in angle_list for scale in scale_list]
            
            with mp.Pool(num_workers) as pool:
                candidates = pool.map(_compute_ncc_score_parallel, args_list)
            
            return candidates
    
    if use_two_stage:
        # ========== 階段 1：粗搜索 ==========
        # 使用較粗的網格進行快速搜索
        if len(ang_grid) > 1:
            # 每隔一個角度（或更多，取決於網格大小）
            step = max(1, len(ang_grid) // 10)  # 至少保留 10 個角度
            coarse_ang = ang_grid[::step]
        else:
            coarse_ang = ang_grid
        
        if len(scales) > 1:
            # 每隔一個縮放
            step = max(1, len(scales) // 3)  # 至少保留 3 個縮放
            coarse_scales = scales[::step]
        else:
            coarse_scales = scales
        
        # 使用並行化計算粗搜索候選
        coarse_candidates = compute_candidates_parallel(coarse_ang, coarse_scales)
        
        # 排序並保留前 N 個最佳候選
        coarse_candidates.sort(key=lambda x: x[0], reverse=True)
        keep_count = max(1, min(topk * 3, int(len(coarse_candidates) * coarse_ratio)))
        best_coarse = coarse_candidates[:keep_count]
        
        # ========== 階段 2：精細搜索 ==========
        # 在粗搜索的候選附近進行更細的搜索
        candidates = []
        angle_step = (ang_grid[1] - ang_grid[0]) if len(ang_grid) > 1 else np.deg2rad(1)
        scale_step = (scales[1] - scales[0]) if len(scales) > 1 else 0.05
        
        # 收集所有精細搜索的參數組合
        fine_args_list = []
        for sc, x, y, th, s in best_coarse:
            # 在最佳候選附近搜索（±2.5 步）
            ang_near = [a for a in ang_grid 
                       if abs(a - th) <= angle_step * 2.5]
            scales_near = [s_val for s_val in scales 
                          if abs(s_val - s) <= scale_step * 2.5]
            
            # 如果附近沒有候選，使用原始值
            if not ang_near:
                ang_near = [th]
            if not scales_near:
                scales_near = [s]
            
            # 收集參數組合
            for theta in ang_near:
                for scale in scales_near:
                    fine_args_list.append((theta, scale))
        
        # 使用並行化計算精細搜索候選
        if should_parallel and len(fine_args_list) > 10:
            # 並行計算
            args_list = [(tpl, img, center_x, center_y, theta, scale, mask)
                        for theta, scale in fine_args_list]
            
            with mp.Pool(num_workers) as pool:
                candidates = pool.map(_compute_ncc_score_parallel, args_list)
        else:
            # 串行計算
            for theta, scale in fine_args_list:
                score = ncc_score_rotated(tpl, img, center_x, center_y, theta, scale, mask)
                candidates.append((score, center_x, center_y, theta, scale))
        
        # 去重（可能粗搜索和精細搜索有重疊）
        seen = set()
        unique_candidates = []
        for cand in candidates:
            key = (round(cand[2], 1), round(cand[3], 1), round(cand[4], 3))  # x, y, theta, scale
            if key not in seen:
                seen.add(key)
                unique_candidates.append(cand)
        
        candidates = unique_candidates
    else:
        # ========== 原始方法：完整搜索 ==========
        candidates = compute_candidates_parallel(ang_grid, scales)
    
    # 排序並返回前 topk
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 調整座標（如果使用了 ROI）
    result = []
    for sc, x, y, th, s in candidates[:topk]:
        if roi is not None:
            x += rx
            y += ry
        result.append((sc, x, y, th, s))
    
    return result


def pyramid_ncc_search_rotated(tpl, img, center_x, center_y, 
                                ang_range=(-np.pi, np.pi), scale_range=(0.7, 1.4),
                                num_levels=4, scale=0.5, mask=None, roi=None,
                                ang_steps=37, scale_steps=15):
    """使用金字塔進行旋轉縮放 NCC 匹配。
    
    Args:
        tpl: 模板圖像
        img: 場景圖像
        center_x, center_y: 初始模板中心位置
        ang_range: 角度搜索範圍 (min_ang, max_ang) 弧度
        scale_range: 縮放搜索範圍 (min_scale, max_scale)
        num_levels: 金字塔層數
        scale: 金字塔縮放因子
        mask: 可選的模板 mask
        roi: 可選的搜索區域限制
        ang_steps: 角度搜索步數
        scale_steps: 縮放搜索步數
    
    Returns:
        (x, y, theta, scale, score): 最佳匹配結果
    """
    tpl = _to_gray(tpl)
    img = _to_gray(img)
    
    # 應用 ROI
    if roi is not None:
        rx, ry, rw, rh = roi
        img = img[ry:ry+rh, rx:rx+rw]
        center_x -= rx
        center_y -= ry
    
    # 構建金字塔
    tpl_pyr = build_pyramid(tpl, num_levels=num_levels, scale=scale)
    img_pyr = build_pyramid(img, num_levels=num_levels, scale=scale)
    
    if mask is not None:
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        mask_pyr = build_pyramid(mask, num_levels=num_levels, scale=scale)
    else:
        mask_pyr = [None] * num_levels
    
    # 從最粗層開始
    # 初始座標需要縮小到最粗層的尺寸
    scale_factor = scale ** (num_levels - 1)
    cx = center_x * scale_factor
    cy = center_y * scale_factor
    best_theta = 0.0
    best_scale = 1.0
    best_score = 0.0
    
    for lvl in range(num_levels-1, -1, -1):
        tpl_l = tpl_pyr[lvl]
        img_l = img_pyr[lvl]
        mask_l = mask_pyr[lvl]
        
        # 在當前層進行搜索
        if lvl == num_levels - 1:
            # 最粗層：全範圍搜索
            ang_grid = np.linspace(ang_range[0], ang_range[1], ang_steps)
            scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
        else:
            # 精細層：在最佳結果附近縮小範圍
            ang_span = (ang_range[1] - ang_range[0]) / (num_levels - lvl)
            scale_span = (scale_range[1] - scale_range[0]) / (num_levels - lvl)
            ang_grid = np.linspace(max(ang_range[0], best_theta - ang_span/2), 
                                 min(ang_range[1], best_theta + ang_span/2), ang_steps)
            scales = np.linspace(max(scale_range[0], best_scale - scale_span/2),
                               min(scale_range[1], best_scale + scale_span/2), scale_steps)
        
        # 網格搜索
        candidates = grid_search_theta_scale_ncc(
            tpl_l, img_l, cx, cy, ang_grid, scales, topk=1, mask=mask_l
        )
        
        if candidates:
            best_score, cx, cy, best_theta, best_scale = candidates[0]
        
        # 縮放座標（從粗到細，座標需要放大到下一層）
        # 注意：最後一層（lvl=0）不需要再放大
        if lvl > 0:
            cx *= (1.0 / scale)  # 因為金字塔是縮小的，所以反向放大
            cy *= (1.0 / scale)
    
    # 轉換回原始座標
    if roi is not None:
        cx += rx
        cy += ry
    
    return (cx, cy, best_theta, best_scale, best_score)


def nms_rotated_matches(matches, min_dist_px=50, min_angle_diff=np.deg2rad(5), 
                        min_scale_diff=0.05, use_position_priority=True):
    """非極大值抑制（NMS）用於旋轉匹配結果，排除重複定位。
    
    Args:
        matches: List of (score, x, y, theta, scale) tuples
        min_dist_px: 最小位置距離（像素），低於此距離視為重複
        min_angle_diff: 最小角度差異（弧度），低於此差異視為重複
        min_scale_diff: 最小縮放差異，低於此差異視為重複
        use_position_priority: 如果 True，位置接近時優先抑制（即使角度/縮放不同）
                            這可以避免同一物體的不同變換被視為多個匹配
    
    Returns:
        List of (score, x, y, theta, scale) tuples after NMS
    """
    if not matches:
        return []
    
    # 按分數降序排序
    matches = sorted(matches, key=lambda x: x[0], reverse=True)
    kept = []
    suppressed = set()
    
    for i, (score_i, x_i, y_i, theta_i, scale_i) in enumerate(matches):
        if i in suppressed:
            continue
        
        kept.append((score_i, x_i, y_i, theta_i, scale_i))
        
        # 檢查與後續匹配的距離
        for j in range(i + 1, len(matches)):
            if j in suppressed:
                continue
            
            score_j, x_j, y_j, theta_j, scale_j = matches[j]
            
            # 計算位置距離
            pos_dist = np.hypot(x_j - x_i, y_j - y_i)
            
            # 計算角度差異（考慮角度週期性）
            angle_diff = abs(theta_j - theta_i)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # 處理 -pi 到 pi 的邊界
            
            # 計算縮放差異
            scale_diff = abs(scale_j - scale_i)
            
            # 改進的判斷邏輯
            if use_position_priority:
                # 方案 1：位置優先 - 如果位置很接近，即使角度/縮放不同也視為重複
                # 這適用於同一個物體的不同匹配實例
                if pos_dist < min_dist_px:
                    # 位置很接近，視為同一個物體，直接抑制
                    suppressed.add(j)
                elif pos_dist < min_dist_px * 1.5:
                    # 位置較接近，且角度和縮放也接近，視為重複
                    if angle_diff < min_angle_diff and scale_diff < min_scale_diff:
                        suppressed.add(j)
            else:
                # 方案 2：原始邏輯 - 三個維度都接近才視為重複
                if pos_dist < min_dist_px and angle_diff < min_angle_diff and scale_diff < min_scale_diff:
                    suppressed.add(j)
    
    return kept


def multi_match_ncc_rotated(tpl, img, initial_positions, ang_grid, scales, 
                            topk_per_position=3, final_topk=10, 
                            min_dist_px=50, min_angle_diff=np.deg2rad(5), 
                            min_scale_diff=0.05, mask=None, roi=None,
                            use_position_priority=True, coarse_first=True,
                            use_parallel=False, num_workers=None,
                            refine_local=False, refine_threshold=0.8,
                            refine_iterative=False, refine_iter_max_iters=5, refine_iter_tol=1e-3):
    """在多個初始位置進行 NCC 旋轉匹配，返回多個不重複的匹配結果。
    
    Args:
        tpl: 模板圖像
        img: 場景圖像
        initial_positions: List of (x, y) 初始位置列表
        ang_grid: 角度網格（弧度）
        scales: 縮放因子列表
        topk_per_position: 每個位置返回的最佳候選數
        final_topk: 最終返回的總匹配數
        min_dist_px: NMS 最小位置距離
        min_angle_diff: NMS 最小角度差異（弧度）
        min_scale_diff: NMS 最小縮放差異
        mask: 可選的模板 mask
        roi: 可選的搜索區域限制
        use_position_priority: 是否使用位置優先的 NMS
        coarse_first: 是否先進行粗搜索
        use_parallel: 是否使用並行化計算（默認 False）
        num_workers: 並行化時的工作進程數（None 表示自動檢測 CPU 核心數）
        refine_local: 是否對匹配結果進行局部精細優化（默認 False）
        refine_threshold: 精細優化的分數閾值，只有分數高於此值的匹配才進行優化
        refine_iterative: 是否對匹配結果進行迭代優化（默認 False，比局部優化更精確但更慢）
        refine_iter_max_iters: 迭代優化的最大迭代次數
        refine_iter_tol: 迭代優化的收斂容差
    
    Returns:
        List of (score, x, y, theta, scale) tuples, sorted by score descending
    """
    all_candidates = []
    
    # 對每個初始位置進行搜索
    for center_x, center_y in initial_positions:
        candidates = grid_search_theta_scale_ncc(
            tpl, img, center_x, center_y, ang_grid, scales, 
            topk=topk_per_position, mask=mask, roi=roi,
            coarse_first=coarse_first,
            use_parallel=use_parallel,
            num_workers=num_workers
        )
        all_candidates.extend(candidates)
    
    # 應用 NMS 排除重複匹配
    matches = nms_rotated_matches(
        all_candidates, 
        min_dist_px=min_dist_px,
        min_angle_diff=min_angle_diff,
        min_scale_diff=min_scale_diff,
        use_position_priority=use_position_priority
    )
    
    # 局部精細優化或迭代優化（如果啟用）
    if refine_iterative:
        # 迭代優化（最精確，但較慢）
        refined_matches = []
        for sc, x, y, th, s in matches[:final_topk]:
            if sc >= refine_threshold:
                # 對高分匹配進行迭代優化
                refined = refine_match_iterative(
                    tpl, img, x, y, th, s,
                    max_iters=refine_iter_max_iters,
                    tol=refine_iter_tol,
                    mask=mask
                )
                refined_matches.append(refined)
            else:
                refined_matches.append((sc, x, y, th, s))
        
        # 重新排序（優化後分數可能改變）
        refined_matches.sort(key=lambda x: x[0], reverse=True)
        return refined_matches[:final_topk]
    elif refine_local:
        # 局部精細優化（快速版本）
        refined_matches = []
        for sc, x, y, th, s in matches[:final_topk]:
            if sc >= refine_threshold:
                # 對高分匹配進行精細優化
                refined = refine_match_local(
                    tpl, img, x, y, th, s,
                    ang_range=np.deg2rad(0.5),  # ±0.5 度
                    scale_range=0.02,  # ±0.02
                    pos_range=2.0,  # ±2 像素
                    ang_steps=11,
                    scale_steps=5,
                    pos_steps=5,
                    mask=mask
                )
                refined_matches.append(refined)
            else:
                refined_matches.append((sc, x, y, th, s))
        
        # 重新排序（精細優化後分數可能改變）
        refined_matches.sort(key=lambda x: x[0], reverse=True)
        return refined_matches[:final_topk]
    else:
        # 返回前 final_topk 個
        return matches[:final_topk]


def multi_match_ncc_from_peaks(tpl, img, scoremap, peaks_or_oxoy, K=10, min_dist_px=50,
                                ang_grid=None, scales=None, topk_per_position=3,
                                min_angle_diff=np.deg2rad(5), min_scale_diff=0.05,
                                mask=None, roi=None, score_threshold=0.7,
                                use_position_priority=True, fast_mode=True,
                                use_parallel=False, num_workers=None,
                                refine_local=False, refine_threshold=0.8,
                                refine_iterative=False, refine_iter_max_iters=5, refine_iter_tol=1e-3):
    """從分數圖的峰值開始，進行多個 NCC 旋轉匹配。
    
    這是一個方便函數，結合了峰值檢測和多匹配搜索。
    支持快速模式：使用金字塔進行粗搜索，快速篩選候選位置。
    
    Args:
        tpl: 模板圖像
        img: 場景圖像
        scoremap: 2D 分數圖（例如從 chamfer_coarse 或 ncc_scoremap 獲得）
        peaks_or_oxoy: 如果是 peaks 列表，則直接使用；如果是 (ox, oy) 元組，則從 scoremap 提取峰值
        K: 從分數圖中提取的峰值數量（僅當 peaks_or_oxoy 是元組時使用）
        min_dist_px: 峰值間最小距離（像素）
        ang_grid: 角度網格（弧度），如果為 None 則使用默認值
        scales: 縮放因子列表，如果為 None 則使用默認值
        topk_per_position: 每個位置返回的最佳候選數
        min_angle_diff: NMS 最小角度差異（弧度）
        min_scale_diff: NMS 最小縮放差異
        mask: 可選的模板 mask
        roi: 可選的搜索區域限制
        score_threshold: 分數閾值，低於此值的匹配會被過濾
        use_position_priority: 是否使用位置優先的 NMS
        fast_mode: 是否啟用快速模式（使用金字塔粗搜索）
        use_parallel: 是否使用並行化計算（默認 False）
        num_workers: 並行化時的工作進程數（None 表示自動檢測 CPU 核心數）
        refine_local: 是否對匹配結果進行局部精細優化（默認 False）
        refine_threshold: 精細優化的分數閾值，只有分數高於此值的匹配才進行優化
        refine_iterative: 是否對匹配結果進行迭代優化（默認 False，比局部優化更精確但更慢）
        refine_iter_max_iters: 迭代優化的最大迭代次數
        refine_iter_tol: 迭代優化的收斂容差
    
    Returns:
        List of (score, x, y, theta, scale) tuples, sorted by score descending
    """
    from .shape_match import topk_peaks
    
    # 判斷 peaks_or_oxoy 是 peaks 還是 (ox, oy)
    if isinstance(peaks_or_oxoy, tuple) and len(peaks_or_oxoy) == 2:
        # 是 (ox, oy)，需要從 scoremap 提取峰值
        ox, oy = peaks_or_oxoy
        peaks = topk_peaks(scoremap, K=K, min_dist=min_dist_px, subpixel=True)
    else:
        # 是 peaks 列表，直接使用
        peaks = peaks_or_oxoy
        ox, oy = 0, 0
    
    # 計算模板中心位置的偏移（從 scoremap 座標轉換到圖像座標）
    tpl_h, tpl_w = tpl.shape[:2]
    
    # 轉換峰值位置為模板中心位置
    initial_positions = []
    for px, py, sc in peaks:
        if sc < score_threshold:
            continue
        # 從 scoremap 座標轉換到圖像座標
        # scoremap[py, px] 對應圖像位置 (px + tpl_w//2 + ox, py + tpl_h//2 + oy)
        center_x = px + tpl_w // 2 + ox
        center_y = py + tpl_h // 2 + oy
        initial_positions.append((center_x, center_y))
    
    # 使用默認參數
    if ang_grid is None:
        ang_grid = np.deg2rad(np.linspace(-180, 180, 37))
    if scales is None:
        scales = np.linspace(0.7, 1.4, 15)
    
    # ========== 快速模式：使用金字塔粗搜索 ==========
    if fast_mode and len(initial_positions) > 3:
        # 對每個初始位置，先用較粗的網格進行快速搜索
        quick_candidates = []
        
        # 使用較粗的網格進行快速搜索
        if len(ang_grid) > 5:
            # 每隔 3 個角度（或更多）
            quick_ang_step = max(1, len(ang_grid) // 5)
            quick_ang = ang_grid[::quick_ang_step]
        else:
            quick_ang = ang_grid
        
        if len(scales) > 2:
            # 每隔 2 個縮放
            quick_scale_step = max(1, len(scales) // 2)
            quick_scales = scales[::quick_scale_step]
        else:
            quick_scales = scales
        
        print(f"[快速模式] 粗搜索: {len(quick_ang)} 角度 × {len(quick_scales)} 縮放 = {len(quick_ang) * len(quick_scales)} 次計算/位置")
        
        # 對每個初始位置進行快速粗搜索
        for cx, cy in initial_positions:
            quick_result = grid_search_theta_scale_ncc(
                tpl, img, cx, cy, quick_ang, quick_scales,
                topk=1,  # 每個位置只保留最佳候選
                mask=mask, 
                roi=roi,
                coarse_first=False,  # 快速模式已經很粗了，不需要再兩階段
                use_parallel=use_parallel,  # 傳遞並行化參數
                num_workers=num_workers
            )
            if quick_result:
                quick_candidates.extend(quick_result)
        
        # 按分數排序，保留前 K 個位置進行精細搜索
        quick_candidates.sort(key=lambda x: x[0], reverse=True)
        refined_positions = [(x, y) for _, x, y, _, _ in quick_candidates[:K]]
        
        print(f"[快速模式] 從 {len(initial_positions)} 個位置篩選到 {len(refined_positions)} 個位置進行精細搜索")
        
        # 使用精細網格對篩選後的位置進行精細搜索（啟用兩階段搜索）
        return multi_match_ncc_rotated(
            tpl, img, refined_positions, ang_grid, scales,
            topk_per_position=topk_per_position,
            final_topk=K,
            min_dist_px=min_dist_px,
            min_angle_diff=min_angle_diff,
            min_scale_diff=min_scale_diff,
            mask=mask,
            roi=roi,
            use_position_priority=use_position_priority,
            coarse_first=True,  # 精細搜索時啟用兩階段搜索
            use_parallel=use_parallel,  # 傳遞並行化參數
            num_workers=num_workers,
            refine_local=refine_local,  # 傳遞精細優化參數
            refine_threshold=refine_threshold,
            refine_iterative=refine_iterative,  # 傳遞迭代優化參數
            refine_iter_max_iters=refine_iter_max_iters,
            refine_iter_tol=refine_iter_tol
        )
    else:
        # ========== 標準模式：完整搜索（也啟用兩階段搜索） ==========
        return multi_match_ncc_rotated(
            tpl, img, initial_positions, ang_grid, scales,
            topk_per_position=topk_per_position,
            final_topk=K,
            min_dist_px=min_dist_px,
            min_angle_diff=min_angle_diff,
            min_scale_diff=min_scale_diff,
            mask=mask,
            roi=roi,
            use_position_priority=use_position_priority,
            coarse_first=True,  # 標準模式也啟用兩階段搜索
            use_parallel=use_parallel,  # 傳遞並行化參數
            num_workers=num_workers,
            refine_local=refine_local,  # 傳遞精細優化參數
            refine_threshold=refine_threshold,
            refine_iterative=refine_iterative,  # 傳遞迭代優化參數
            refine_iter_max_iters=refine_iter_max_iters,
            refine_iter_tol=refine_iter_tol
        )
