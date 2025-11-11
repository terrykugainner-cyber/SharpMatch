"""
仿射匹配模組：使用特徵檢測（SIFT/ORB/AKAZE）+ RANSAC 估計仿射變換
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from .matcher import match_features as match_features_with_matcher


def detect_features(img: np.ndarray, detector_type: str = 'auto', max_keypoints: int = 3000) -> Tuple[str, List, np.ndarray]:
    """檢測圖像特徵點和描述子。
    
    Args:
        img: 輸入圖像（灰度或 BGR）
        detector_type: 檢測器類型 ('auto', 'sift', 'orb', 'akaze')
        max_keypoints: 最多特徵點數量
    
    Returns:
        (actual_detector_type, keypoints, descriptors)
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 自動選擇檢測器
    actual_detector_type = detector_type
    if detector_type == 'auto':
        # 檢查 SIFT 是否可用
        try:
            detector = cv2.SIFT_create(nfeatures=max_keypoints)
            actual_detector_type = 'sift'
        except:
            # SIFT 不可用，使用 ORB
            detector = cv2.ORB_create(nfeatures=max_keypoints)
            actual_detector_type = 'orb'
    elif detector_type == 'sift':
        try:
            detector = cv2.SIFT_create(nfeatures=max_keypoints)
            actual_detector_type = 'sift'
        except:
            raise ValueError("SIFT 檢測器不可用，請檢查 OpenCV 版本（需要 4.5.1+）或使用 'orb' 或 'akaze'")
    elif detector_type == 'orb':
        detector = cv2.ORB_create(nfeatures=max_keypoints)
        actual_detector_type = 'orb'
    elif detector_type == 'akaze':
        # AKAZE 不支持 nfeatures 參數，需要在檢測後手動限制
        detector = cv2.AKAZE_create()
        actual_detector_type = 'akaze'
    else:
        raise ValueError(f"不支援的檢測器類型: {detector_type}")
    
    # 檢測特徵點和描述子
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    if descriptors is None or len(keypoints) == 0:
        raise ValueError(f"未檢測到特徵點。建議：1) 使用更大的圖像 2) 調整檢測器參數 3) 確保圖像清晰")
    
    # 確保 keypoints 和 descriptors 數量一致
    if len(keypoints) != len(descriptors):
        raise ValueError(f"關鍵點和描述子數量不一致: {len(keypoints)} vs {len(descriptors)}")
    
    # 對於 AKAZE（不支持 nfeatures 參數），在檢測後手動限制特徵點數量
    if actual_detector_type == 'akaze' and len(keypoints) > max_keypoints:
        # 按響應值排序，保留前 max_keypoints 個
        # 創建 (keypoint, descriptor) 配對並排序
        kp_desc_pairs = list(zip(keypoints, descriptors))
        kp_desc_pairs.sort(key=lambda x: x[0].response, reverse=True)
        kp_desc_pairs = kp_desc_pairs[:max_keypoints]
        # 分離 keypoints 和 descriptors
        keypoints = [kp for kp, _ in kp_desc_pairs]
        descriptors = np.array([desc for _, desc in kp_desc_pairs])
        # 確保描述子是 2D 數組
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)
    
    return actual_detector_type, keypoints, descriptors


def match_features(desc1: np.ndarray, desc2: np.ndarray, 
                   detector_type: str, 
                   matcher_type: str = 'bf',
                   ratio_threshold: float = 0.75) -> List[cv2.DMatch]:
    """使用 KNN 匹配（k=2）和 Lowe Ratio Test 進行特徵匹配。
    
    Args:
        desc1: 模板描述子
        desc2: 場景描述子
        detector_type: 檢測器類型（用於選擇距離度量）
        matcher_type: 匹配器類型 ('bf' 或 'flann')，預設 'bf'
        ratio_threshold: Lowe Ratio Test 閾值
    
    Returns:
        匹配列表
    """
    return match_features_with_matcher(
        desc1, desc2, detector_type, matcher_type, ratio_threshold
    )


def estimate_affine_partial(kp1: List, kp2: List, matches: List[cv2.DMatch],
                            ransac_reproj_thresh: float = 3.0,
                            ransac_max_iters: int = 2000,
                            ransac_confidence: float = 0.99) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """使用 RANSAC 估計相似仿射變換（平移+旋轉+等比縮放）。
    
    Args:
        kp1: 模板關鍵點列表
        kp2: 場景關鍵點列表
        matches: 匹配列表
        ransac_reproj_thresh: RANSAC 重投影閾值（像素）
        ransac_max_iters: RANSAC 最大迭代次數
        ransac_confidence: RANSAC 置信度
    
    Returns:
        (affine_matrix, inliers_mask) 或 (None, None) 如果失敗
    """
    if len(matches) < 3:
        raise ValueError(f"匹配點太少（{len(matches)}），至少需要 3 個點。建議：1) 放寬 ratio 閾值 2) 增加特徵點數量")
    
    # 提取匹配點座標
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    # 如果使用了 ROI，需要調整場景點的座標
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # 使用 estimateAffinePartial2D 估計相似仿射（6 個參數）
    # 這會估計平移、旋轉、等比縮放，並抑制斜切
    affine_matrix, inliers_mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_thresh,
        maxIters=ransac_max_iters,
        confidence=ransac_confidence,
        refineIters=10
    )
    
    if affine_matrix is None:
        raise ValueError("無法估計仿射矩陣。建議：1) 增加匹配點數 2) 放寬 RANSAC 參數 3) 檢查圖像對應關係")
    
    # inliers_mask 是布林向量
    inliers_mask = inliers_mask.ravel().astype(bool)
    
    return affine_matrix, inliers_mask


def decompose_affine_matrix(M: np.ndarray) -> Dict[str, float]:
    """從仿射矩陣分解出平移、旋轉角度和縮放因子。
    
    仿射矩陣 M 的格式：
    [a  b  tx]
    [c  d  ty]
    
    對於相似變換（無斜切），有：
    a = s * cos(θ)
    b = -s * sin(θ)
    c = s * sin(θ)
    d = s * cos(θ)
    
    Args:
        M: 2x3 仿射矩陣
    
    Returns:
        包含 tx, ty, theta_deg, scale 的字典
    """
    if M is None or M.shape != (2, 3):
        raise ValueError("無效的仿射矩陣")
    
    # 提取平移
    tx = float(M[0, 2])
    ty = float(M[1, 2])
    
    # 提取旋轉和縮放部分
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    
    # 計算縮放因子（取平均以提高穩定性）
    scale_x = np.sqrt(a * a + c * c)
    scale_y = np.sqrt(b * b + d * d)
    scale = (scale_x + scale_y) / 2.0
    
    # 計算旋轉角度（使用 atan2 確保正確象限）
    # 從 a = s*cos(θ), c = s*sin(θ) 可以得到 θ
    theta_rad = np.arctan2(c, a)
    theta_deg = np.rad2deg(theta_rad)
    
    return {
        'tx': tx,
        'ty': ty,
        'theta_deg': theta_deg,
        'theta_rad': theta_rad,
        'scale': scale,
        'scale_x': scale_x,
        'scale_y': scale_y
    }


def diagnose_feature_quality(num_keypoints_template: int,
                             num_keypoints_scene: int,
                             num_ratio_matches: int,
                             detector_type: str,
                             ratio_threshold: float,
                             max_keypoints: int,
                             min_keypoints_scene: int = 30,
                             min_ratio_matches: int = 15) -> List[str]:
    """檢查特徵點與匹配數量是否偏低，並提供診斷建議。"""
    warnings = []

    if num_keypoints_scene < min_keypoints_scene:
        suggestions = [
            f"場景特徵點僅 {num_keypoints_scene} 個 (< {min_keypoints_scene})",
            f"- 建議提高 max_keypoints（目前 {max_keypoints}）",
            "- 優先選用 SIFT/高品質檢測器以取得更多特徵",
            "- 嘗試提升影像對比、銳化或降噪以強化特徵"
        ]
        warnings.append("\n".join(suggestions))

    if num_ratio_matches < min_ratio_matches:
        suggestions = [
            f"通過 ratio test 的匹配僅 {num_ratio_matches} 個 (< {min_ratio_matches})",
            f"- 建議放寬 ratio_threshold（目前 {ratio_threshold:.2f}）",
            f"- 同時提高 max_keypoints（目前 {max_keypoints}）",
            "- 針對場景/模板提升對比、邊緣或清晰度"
        ]
        if detector_type.lower() != 'sift':
            suggestions.append("- 可改用 SIFT 以獲得更穩定的描述子")
        warnings.append("\n".join(suggestions))

    return warnings


def affine_match(template: np.ndarray, image: np.ndarray,
                detector_type: str = 'auto',
                matcher_type: str = 'bf',
                ratio_threshold: float = 0.75,
                max_keypoints: int = 3000,
                ransac_reproj_thresh: float = 3.0,
                ransac_max_iters: int = 2000,
                ransac_confidence: float = 0.99,
                roi: Optional[Tuple[int, int, int, int]] = None) -> Dict:
    """完整的仿射匹配流程。
    
    Args:
        template: 模板圖像
        image: 場景圖像
        detector_type: 檢測器類型
        matcher_type: 匹配器類型 ('bf' 或 'flann')，預設 'bf'
        ratio_threshold: Lowe Ratio Test 閾值
        max_keypoints: 最多特徵點數量
        ransac_reproj_thresh: RANSAC 重投影閾值
        ransac_max_iters: RANSAC 最大迭代次數
        ransac_confidence: RANSAC 置信度
        roi: 可選的搜索區域 (x, y, w, h)，如果提供則只在該區域搜索
    
    Returns:
        包含所有匹配結果的字典
    """
    # 如果提供了 ROI，裁剪圖像（但需要調整座標）
    image_roi = image
    roi_offset_x = 0
    roi_offset_y = 0
    
    if roi is not None:
        x, y, w, h = roi
        # 確保 ROI 在圖像範圍內
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w > 0 and h > 0:
            image_roi = image[y:y+h, x:x+w]
            roi_offset_x = x
            roi_offset_y = y
        else:
            raise ValueError(f"無效的 ROI: ({x}, {y}, {w}, {h})")
    
    # 檢測特徵（實際使用的檢測器類型可能會改變，例如 auto 模式）
    actual_detector_type1, kp1, desc1 = detect_features(template, detector_type, max_keypoints)
    actual_detector_type2, kp2, desc2 = detect_features(image_roi, detector_type, max_keypoints)
    
    # 確保兩個圖像使用相同的檢測器類型
    if actual_detector_type1 != actual_detector_type2:
        raise ValueError(f"檢測器類型不一致: 模板使用 {actual_detector_type1}，場景使用 {actual_detector_type2}")
    
    actual_detector_type = actual_detector_type1
    
    # 匹配特徵
    matches = match_features(desc1, desc2, actual_detector_type, matcher_type, ratio_threshold)

    # 例外處理與診斷
    diagnostics = diagnose_feature_quality(
        num_keypoints_template=len(kp1),
        num_keypoints_scene=len(kp2),
        num_ratio_matches=len(matches),
        detector_type=actual_detector_type,
        ratio_threshold=ratio_threshold,
        max_keypoints=max_keypoints
    )
    if diagnostics:
        print("\n[診斷] 特徵/匹配數量可能偏低，建議調整：")
        for msg in diagnostics:
            lines = msg.split('\n')
            print(f"  - {lines[0]}")
            for extra in lines[1:]:
                print(f"    {extra}")

    if len(matches) < 3:
        raise ValueError(f"匹配點太少（{len(matches)}），至少需要 3 個點。建議：1) 放寬 ratio 閾值（當前 {ratio_threshold}） 2) 增加特徵點數量（當前 {max_keypoints}） 3) 確保模板和場景圖像有足夠的重疊區域")
    
    # 估計仿射變換
    affine_matrix, inliers_mask = estimate_affine_partial(
        kp1, kp2, matches,
        ransac_reproj_thresh, ransac_max_iters, ransac_confidence
    )
    
    # 分解參數
    params = decompose_affine_matrix(affine_matrix)
    
    # 如果使用了 ROI，需要調整仿射矩陣的平移部分和關鍵點座標
    if roi is not None:
        affine_matrix[0, 2] += roi_offset_x
        affine_matrix[1, 2] += roi_offset_y
        # 重新分解參數（因為平移改變了）
        params = decompose_affine_matrix(affine_matrix)
        
        # 調整關鍵點座標到原始圖像座標系
        kp2_adjusted = []
        for kp in kp2:
            kp2_adjusted.append(cv2.KeyPoint(
                kp.pt[0] + roi_offset_x,
                kp.pt[1] + roi_offset_y,
                kp.size, kp.angle, kp.response, kp.octave, kp.class_id
            ))
        kp2 = kp2_adjusted
    
    # 統計內點數量
    num_inliers = int(np.sum(inliers_mask))
    num_outliers = len(matches) - num_inliers
    inlier_ratio = num_inliers / len(matches) if len(matches) > 0 else 0.0
    
    return {
        'affine_matrix': affine_matrix,
        'inliers_mask': inliers_mask,
        'matches': matches,
        'keypoints_template': kp1,
        'keypoints_image': kp2,
        'detector_type': actual_detector_type,
        'params': params,
        'num_matches': len(matches),
        'num_inliers': num_inliers,
        'num_outliers': num_outliers,
        'inlier_ratio': inlier_ratio,
        'ratio_threshold': ratio_threshold,
        'matcher_type': matcher_type,
        'ransac_params': {
            'reproj_thresh': ransac_reproj_thresh,
            'max_iters': ransac_max_iters,
            'confidence': ransac_confidence
        },
        'diag_warnings': diagnostics,
    }


def visualize_matches(template: np.ndarray, image: np.ndarray,
                      result: Dict, 
                      show_all: bool = False) -> np.ndarray:
    """繪製匹配可視化圖（綠色內點、紅色外點）。
    
    Args:
        template: 模板圖像
        image: 場景圖像
        result: affine_match 返回的結果字典
        show_all: 是否顯示所有匹配點（True）或僅顯示內點（False）
    
    Returns:
        可視化圖像
    """
    kp1 = result['keypoints_template']
    kp2 = result['keypoints_image']
    matches = result['matches']
    inliers_mask = result['inliers_mask']
    
    # 轉換為彩色圖像
    if template.ndim == 2:
        template_color = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    else:
        template_color = template.copy()
    
    if image.ndim == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()
    
    # 創建拼接圖像
    h1, w1 = template_color.shape[:2]
    h2, w2 = image_color.shape[:2]
    vis_h = max(h1, h2)
    vis_w = w1 + w2
    
    vis = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    vis[:h1, :w1] = template_color
    vis[:h2, w1:w1+w2] = image_color
    
    # 調整關鍵點座標（場景圖像的 x 座標需要加上模板寬度）
    kp2_shifted = []
    for kp in kp2:
        kp2_shifted.append(cv2.KeyPoint(kp.pt[0] + w1, kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id))
    
    # 繪製匹配線
    for i, match in enumerate(matches):
        if not show_all and not inliers_mask[i]:
            continue
        
        pt1 = tuple(map(int, kp1[match.queryIdx].pt))
        pt2 = tuple(map(int, kp2_shifted[match.trainIdx].pt))
        
        # 內點用綠色，外點用紅色
        if inliers_mask[i]:
            color = (0, 255, 0)  # 綠色
            thickness = 2
        else:
            color = (0, 0, 255)  # 紅色
            thickness = 1
        
        cv2.line(vis, pt1, pt2, color, thickness)
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.circle(vis, pt2, 3, color, -1)
    
    # 添加文字信息
    info_text = f"Matches: {len(matches)}, Inliers: {result['num_inliers']}, Ratio: {result['inlier_ratio']:.2%}"
    cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis


def visualize_reprojection(template: np.ndarray, image: np.ndarray,
                          result: Dict) -> np.ndarray:
    """繪製模板外框重投影到場景圖像上。
    
    Args:
        template: 模板圖像
        image: 場景圖像
        result: affine_match 返回的結果字典
    
    Returns:
        重投影可視化圖像
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    M = result['affine_matrix']
    if M is None:
        return vis
    
    # 模板框的四個角點（在模板座標系中）
    h, w = template.shape[:2]
    corners_template = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ]).reshape(-1, 1, 2)
    
    # 應用仿射變換
    corners_projected = cv2.transform(corners_template, M)
    corners_projected = corners_projected.reshape(-1, 2).astype(np.int32)
    
    # 繪製投影框
    cv2.polylines(vis, [corners_projected], True, (0, 255, 0), 2, cv2.LINE_AA)
    
    # 繪製角點
    for i, corner in enumerate(corners_projected):
        cv2.circle(vis, tuple(corner), 5, (0, 255, 0), -1)
        cv2.putText(vis, f'P{i+1}', (corner[0] + 5, corner[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 添加參數信息
    params = result['params']
    info_lines = [
        f"Translation: ({params['tx']:.1f}, {params['ty']:.1f})",
        f"Rotation: {params['theta_deg']:.1f} deg",
        f"Scale: {params['scale']:.3f}",
        f"Inliers: {result['num_inliers']}/{result['num_matches']}"
    ]
    
    y_offset = 20
    for line in info_lines:
        cv2.putText(vis, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
    
    return vis


def nms_affine_matches(results: List[Dict], 
                      min_distance: float = 50.0,
                      min_angle_diff: float = 5.0,
                      min_scale_diff: float = 0.05) -> List[Dict]:
    """使用非極大值抑制（NMS）去除重複的仿射匹配結果。
    
    Args:
        results: 匹配結果列表
        min_distance: 最小位置距離（像素），低於此距離視為重複
        min_angle_diff: 最小角度差異（度），低於此差異視為重複
        min_scale_diff: 最小縮放差異，低於此差異視為重複
    
    Returns:
        去重後的匹配結果列表
    """
    if len(results) == 0:
        return []
    
    # 按內點比率排序（降序）
    results = sorted(results, key=lambda r: r['inlier_ratio'], reverse=True)
    
    kept = []
    suppressed = set()
    
    for i, result_i in enumerate(results):
        if i in suppressed:
            continue
        
        kept.append(result_i)
        params_i = result_i['params']
        
        # 檢查與後續結果的距離
        for j in range(i + 1, len(results)):
            if j in suppressed:
                continue
            
            result_j = results[j]
            params_j = result_j['params']
            
            # 計算位置距離
            pos_dist = np.hypot(params_j['tx'] - params_i['tx'], 
                               params_j['ty'] - params_i['ty'])
            
            # 計算角度差異（考慮週期性）
            angle_diff = abs(params_j['theta_deg'] - params_i['theta_deg'])
            angle_diff = min(angle_diff, 360.0 - angle_diff)
            
            # 計算縮放差異
            scale_diff = abs(params_j['scale'] - params_i['scale'])
            
            # 如果位置、角度、縮放都接近，視為重複
            if (pos_dist < min_distance and 
                angle_diff < min_angle_diff and 
                scale_diff < min_scale_diff):
                suppressed.add(j)
    
    return kept


def multi_affine_match_hybrid(template: np.ndarray, image: np.ndarray,
                              chamfer_k: int = 20,
                              chamfer_min_dist: int = 50,
                              chamfer_roi: Optional[Tuple[int, int, int, int]] = None,
                              min_inlier_ratio: float = 0.15,
                              min_matches: int = 3,
                              min_inliers: Optional[int] = None,
                              scale_range: Optional[Tuple[float, float]] = None,
                              min_chamfer_score: Optional[float] = None,
                              affine_roi_padding: int = 50,
                              detector_type: str = 'auto',
                              matcher_type: str = 'bf',
                              ratio_threshold: float = 0.75,
                              max_keypoints: int = 3000,
                              ransac_reproj_thresh: float = 3.0,
                              ransac_max_iters: int = 2000,
                              ransac_confidence: float = 0.99,
                              nms_min_distance: float = 50.0,
                              nms_min_angle_diff: float = 5.0,
                              nms_min_scale_diff: float = 0.05,
                              post_min_inlier_ratio: Optional[float] = None,
                              post_min_matches: Optional[int] = None,
                              post_min_inliers: Optional[int] = None,
                              post_scale_range: Optional[Tuple[float, float]] = None,
                              post_min_chamfer_score: Optional[float] = None,
                              verbose: bool = False) -> List[Dict]:
    """使用 Chamfer 粗定位 + Affine 細定位進行多物件檢測（方案 3）。
    
    流程：
    1. 使用 Chamfer Distance 在場景中進行粗定位，找到多個候選位置
    2. 對每個候選位置，使用 Affine 匹配進行精細定位
    3. 使用 NMS 去除重複匹配
    4. 可選地於 Stage 2/Stage 3 設定額外的品質門檻
    
    Args:
        template: 模板圖像
        image: 場景圖像
        chamfer_k: Chamfer 檢測的候選位置數量
        chamfer_min_dist: Chamfer 峰值間最小距離
        chamfer_roi: Chamfer 搜索區域 (x, y, w, h)，None 表示全圖
        min_inlier_ratio: Stage 2 所需的最小內點比率閾值
        min_matches: Stage 2 所需的最小匹配點數
        min_inliers: Stage 2 所需的最小內點數（可選）
        scale_range: Stage 2 可接受的縮放範圍 (min_scale, max_scale)（可選）
        min_chamfer_score: Stage 2 所需的最小 Chamfer 分數（可選）
        affine_roi_padding: 在候選位置周圍的 Affine 搜索區域填充（像素）
        detector_type: Affine 檢測器類型
        matcher_type: 匹配器類型 ('bf' 或 'flann')，預設 'bf'
        ratio_threshold: Lowe Ratio Test 閾值
        max_keypoints: 最多特徵點數量
        ransac_reproj_thresh: RANSAC 重投影閾值
        ransac_max_iters: RANSAC 最大迭代次數
        ransac_confidence: RANSAC 置信度
        nms_min_distance: NMS 最小位置距離
        nms_min_angle_diff: NMS 最小角度差異（度）
        nms_min_scale_diff: NMS 最小縮放差異
        post_min_inlier_ratio: NMS 之後的最小內點比率閾值（可選）
        post_min_matches: NMS 之後的最小匹配數（可選）
        post_min_inliers: NMS 之後的最小內點數（可選）
        post_scale_range: NMS 之後的縮放範圍（可選）
        post_min_chamfer_score: NMS 之後的最小 Chamfer 分數（可選）
        verbose: 是否顯示詳細信息
    
    Returns:
        匹配結果列表，每個元素是一個匹配結果字典
    """
    from .shape_model import create
    from .shape_match import chamfer_coarse, topk_peaks
    
    if verbose:
        print(f"[混合匹配] 開始多物件檢測...")
        print(f"  Chamfer 候選數: {chamfer_k}")
        print(f"  最小內點比率: {min_inlier_ratio}")
    
    # ========== 階段 1: Chamfer 粗定位 ==========
    if verbose:
        print(f"\n[階段 1] Chamfer 粗定位...")
    
    # 創建模板的形狀模型
    tpl_h, tpl_w = template.shape[:2]
    model = create(template, (0, 0, tpl_w, tpl_h), 
                   edge_threshold=(60, 140), sampling_step=2, use_polarity=False)
    
    # 執行 Chamfer 匹配
    scoremap, (ox, oy) = chamfer_coarse(model, image, roi=chamfer_roi)
    
    # 提取候選位置
    peaks = topk_peaks(scoremap, K=chamfer_k, min_dist=chamfer_min_dist, subpixel=True)
    
    if verbose:
        print(f"  找到 {len(peaks)} 個候選位置")
    
    if len(peaks) == 0:
        if verbose:
            print("  警告: 未找到任何候選位置")
        return []
    
    # ========== 階段 2: Affine 細定位 ==========
    if verbose:
        print(f"\n[階段 2] Affine 細定位...")
    
    all_results = []
    img_h, img_w = image.shape[:2]
    
    for idx, (px, py, chamfer_score) in enumerate(peaks):
        # 計算候選位置（模板中心）
        center_x = tpl_w // 2 + px + ox
        center_y = tpl_h // 2 + py + oy
        
        # 計算搜索 ROI（在候選位置周圍）
        roi_x = max(0, int(center_x - tpl_w // 2 - affine_roi_padding))
        roi_y = max(0, int(center_y - tpl_h // 2 - affine_roi_padding))
        roi_w = min(img_w - roi_x, tpl_w + 2 * affine_roi_padding)
        roi_h = min(img_h - roi_y, tpl_h + 2 * affine_roi_padding)
        
        if roi_w <= 0 or roi_h <= 0:
            if verbose:
                print(f"  候選 {idx+1}: 位置超出範圍，跳過")
            continue
        
        roi = (roi_x, roi_y, roi_w, roi_h)
        
        if verbose:
            print(f"  候選 {idx+1}/{len(peaks)}: 位置=({center_x:.1f}, {center_y:.1f}), "
                  f"Chamfer 分數={chamfer_score:.4f}")
        
        try:
            # 在 ROI 內執行 Affine 匹配
            result = affine_match(
                template, image,
                detector_type=detector_type,
                matcher_type=matcher_type,
                ratio_threshold=ratio_threshold,
                max_keypoints=max_keypoints,
                ransac_reproj_thresh=ransac_reproj_thresh,
                ransac_max_iters=ransac_max_iters,
                ransac_confidence=ransac_confidence,
                roi=roi
            )
            skip_reasons = []
            if result['inlier_ratio'] < min_inlier_ratio:
                skip_reasons.append(
                    f"內點比率 {result['inlier_ratio']:.2%} 低於閾值 {min_inlier_ratio:.2%}")
            if result['num_matches'] < min_matches:
                skip_reasons.append(
                    f"匹配點數 {result['num_matches']} 少於最小要求 {min_matches}")
            if min_inliers is not None and result['num_inliers'] < min_inliers:
                skip_reasons.append(
                    f"內點數 {result['num_inliers']} 少於最小要求 {min_inliers}")
            scale_value = result['params']['scale']
            if scale_range is not None:
                min_scale, max_scale = scale_range
                if not (min_scale <= scale_value <= max_scale):
                    skip_reasons.append(
                        f"縮放 {scale_value:.3f} 不在允許範圍 [{min_scale}, {max_scale}] 內")
            if min_chamfer_score is not None and chamfer_score < min_chamfer_score:
                skip_reasons.append(
                    f"Chamfer 分數 {chamfer_score:.4f} 低於閾值 {min_chamfer_score}")

            if skip_reasons:
                if verbose:
                    for reason in skip_reasons:
                        print(f"    → [SKIP] {reason}")
                continue
            
            # 添加 Chamfer 分數到結果中
            result['chamfer_score'] = chamfer_score
            result['chamfer_position'] = (px, py)
            
            all_results.append(result)
            
            if verbose:
                params = result['params']
                print(f"    → [OK] 匹配成功: 內點比率={result['inlier_ratio']:.2%}, "
                      f"位置=({params['tx']:.1f}, {params['ty']:.1f}), "
                      f"角度={params['theta_deg']:.1f}°, 縮放={params['scale']:.3f}")
        
        except Exception as e:
            if verbose:
                print(f"    → [FAIL] 匹配失敗: {e}")
            continue
    
    if verbose:
        print(f"\n  Affine 細定位完成: {len(all_results)} 個有效匹配")
    
    # ========== 階段 3: NMS 去重 ==========
    if verbose:
        print(f"\n[階段 3] NMS 去重...")
    
    final_results = nms_affine_matches(
        all_results,
        min_distance=nms_min_distance,
        min_angle_diff=nms_min_angle_diff,
        min_scale_diff=nms_min_scale_diff
    )
    
    if verbose:
        print(f"  NMS 後保留: {len(final_results)} 個匹配")
    
    filtered_results = []
    filtered_out = []
    for res in final_results:
        post_reasons = []
        if post_min_inlier_ratio is not None and res['inlier_ratio'] < post_min_inlier_ratio:
            post_reasons.append(
                f"內點比率 {res['inlier_ratio']:.2%} 低於最終閾值 {post_min_inlier_ratio:.2%}")
        if post_min_matches is not None and res['num_matches'] < post_min_matches:
            post_reasons.append(
                f"匹配點數 {res['num_matches']} 少於最終要求 {post_min_matches}")
        if post_min_inliers is not None and res['num_inliers'] < post_min_inliers:
            post_reasons.append(
                f"內點數 {res['num_inliers']} 少於最終要求 {post_min_inliers}")
        if post_scale_range is not None:
            post_min_scale, post_max_scale = post_scale_range
            s = res['params']['scale']
            if not (post_min_scale <= s <= post_max_scale):
                post_reasons.append(
                    f"縮放 {s:.3f} 不在最終範圍 [{post_min_scale}, {post_max_scale}] 內")
        if post_min_chamfer_score is not None:
            chamfer_val = res.get('chamfer_score')
            if chamfer_val is None or chamfer_val < post_min_chamfer_score:
                post_reasons.append(
                    f"Chamfer 分數 {chamfer_val if chamfer_val is not None else 'None'} "
                    f"低於最終閾值 {post_min_chamfer_score}")
        if post_reasons:
            filtered_out.append((res, post_reasons))
        else:
            filtered_results.append(res)
    
    if verbose and filtered_out:
        print(f"\n[最終過濾] 移除 {len(filtered_out)} 個結果：")
        for res, reasons in filtered_out:
            params = res['params']
            print(f"  - 位置=({params['tx']:.1f}, {params['ty']:.1f}), "
                  f"角度={params['theta_deg']:.1f}°, 縮放={params['scale']:.3f}")
            for reason in reasons:
                print(f"      • {reason}")
    
    return filtered_results


def visualize_multi_matches(template: np.ndarray, image: np.ndarray,
                             results: List[Dict],
                             show_all_matches: bool = False) -> np.ndarray:
    """繪製多個匹配結果的可視化。
    
    Args:
        template: 模板圖像
        image: 場景圖像
        results: 多個匹配結果列表
        show_all_matches: 是否顯示所有匹配點（True）或僅顯示內點（False）
    
    Returns:
        可視化圖像
    """
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    # 定義顏色列表（用於不同實例）
    colors = [
        (0, 255, 0),    # 綠色
        (255, 0, 0),    # 藍色
        (0, 0, 255),    # 紅色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 洋紅色
        (0, 255, 255),  # 黃色
        (128, 0, 128),  # 紫色
        (255, 165, 0),  # 橙色
        (128, 128, 0),  # 橄欖色
        (0, 128, 128),  # 深青色
    ]
    
    # 繪製每個匹配結果
    for idx, result in enumerate(results):
        color = colors[idx % len(colors)]
        
        # 繪製模板框投影
        M = result['affine_matrix']
        if M is not None:
            h, w = template.shape[:2]
            corners_template = np.float32([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ]).reshape(-1, 1, 2)
            
            corners_projected = cv2.transform(corners_template, M)
            corners_projected = corners_projected.reshape(-1, 2).astype(np.int32)
            
            # 繪製投影框
            cv2.polylines(vis, [corners_projected], True, color, 2, cv2.LINE_AA)
            
            # 繪製中心點
            params = result['params']
            center = (int(params['tx']), int(params['ty']))
            cv2.circle(vis, center, 5, color, -1)
            
            # 添加標籤
            label = f"#{idx+1}: {result['inlier_ratio']:.1%}"
            cv2.putText(vis, label, (center[0] + 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 可選：繪製匹配點
        if show_all_matches:
            kp2 = result['keypoints_image']
            matches = result['matches']
            inliers_mask = result['inliers_mask']
            
            for i, match in enumerate(matches):
                pt2 = tuple(map(int, kp2[match.trainIdx].pt))
                if inliers_mask[i]:
                    cv2.circle(vis, pt2, 3, color, -1)
                else:
                    cv2.circle(vis, pt2, 2, (128, 128, 128), -1)
    
    # 添加總體信息
    info_text = f"Found {len(results)} instances"
    cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis


def warp_scene_to_template(template: np.ndarray, 
                          scene: np.ndarray, 
                          affine_matrix: np.ndarray,
                          interpolation: int = cv2.INTER_LINEAR,
                          border_mode: int = cv2.BORDER_CONSTANT,
                          border_value: Union[float, Tuple[float, float, float]] = 0,
                          output_mask: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """根據仿射變換矩陣將場景影像反扭轉到模板大小。
    
    此函式將場景影像中對應模板的區域，根據仿射變換矩陣反扭轉回模板的座標系，
    生成與模板尺寸相同的影像，方便後續 NCC/Chamfer 精修。
    
    Args:
        template: 模板圖像（用於確定輸出尺寸）
        scene: 場景圖像
        affine_matrix: 2x3 仿射變換矩陣（從模板到場景的變換）
        interpolation: 插值方法（預設 cv2.INTER_LINEAR）
        border_mode: 邊界處理模式（預設 cv2.BORDER_CONSTANT）
        border_value: 邊界填充值（預設 0）；彩色圖像可使用長度為 3 的 tuple
        output_mask: 是否輸出有效區域遮罩
    
    Returns:
        如果 output_mask=False: 返回反扭轉後的圖像（與模板同尺寸）
        如果 output_mask=True: 返回 (warped_image, mask) 元組，其中 mask 為二值遮罩，
                               255 表示有效像素（來自場景圖像），0 表示填充像素
    """
    if affine_matrix is None or affine_matrix.shape != (2, 3):
        raise ValueError("無效的仿射矩陣")
    
    tpl_h, tpl_w = template.shape[:2]
    warp_flags = interpolation | cv2.WARP_INVERSE_MAP
    
    # 為彩色圖像準備正確的邊界填充值
    if scene.ndim == 3:
        if isinstance(border_value, (int, float)):
            border_value_actual = tuple([border_value] * scene.shape[2])
        elif isinstance(border_value, (list, tuple, np.ndarray)):
            if len(border_value) != scene.shape[2]:
                raise ValueError("彩色圖像的 border_value 必須與通道數相同")
            border_value_actual = tuple(border_value)
        else:
            raise ValueError("border_value 類型無效，應為數值或長度符合通道數的序列")
    else:
        border_value_actual = float(border_value)
    
    warped = cv2.warpAffine(
        scene,
        affine_matrix,
        (tpl_w, tpl_h),
        flags=warp_flags,
        borderMode=border_mode,
        borderValue=border_value_actual,
    )
    
    if output_mask:
        scene_mask = np.ones(scene.shape[:2], dtype=np.uint8) * 255
        mask = cv2.warpAffine(
            scene_mask,
            affine_matrix,
            (tpl_w, tpl_h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = (mask > 0).astype(np.uint8) * 255
        return warped, mask
    
    return warped


def visualize_warp_comparison(template: np.ndarray, 
                             scene: np.ndarray, 
                             warped_scene: np.ndarray,
                             result: Dict) -> np.ndarray:
    """創建模板、場景和反扭轉後的場景區塊的並排比較圖。
    
    Args:
        template: 模板圖像
        scene: 場景圖像
        warped_scene: 反扭轉後的場景區塊（與模板同尺寸）
        result: affine_match 返回的結果字典
    
    Returns:
        並排比較圖像
    """
    # 確保所有圖像都是相同類型（灰度或彩色）
    def ensure_color(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()
    
    template_color = ensure_color(template)
    warped_color = ensure_color(warped_scene)
    
    # 在場景圖像上繪製檢測框
    scene_color = ensure_color(scene)
    M = result['affine_matrix']
    if M is not None:
        h, w = template.shape[:2]
        corners_template = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ]).reshape(-1, 1, 2)
        
        corners_projected = cv2.transform(corners_template, M)
        corners_projected = corners_projected.reshape(-1, 2).astype(np.int32)
        
        # 繪製投影框
        cv2.polylines(scene_color, [corners_projected], True, (0, 255, 0), 2, cv2.LINE_AA)
    
    # 調整圖像大小以保持一致
    target_h = max(template_color.shape[0], warped_color.shape[0])
    target_w = max(template_color.shape[1], warped_color.shape[1])
    
    template_resized = cv2.resize(template_color, (target_w, target_h))
    warped_resized = cv2.resize(warped_color, (target_w, target_h))
    
    # 調整場景圖像大小，保持寬高比
    scene_h, scene_w = scene_color.shape[:2]
    scene_aspect = scene_w / scene_h
    scene_target_w = int(target_w * 0.8)  # 場景圖像稍小一些以留出空間
    scene_target_h = int(scene_target_w / scene_aspect)
    scene_resized = cv2.resize(scene_color, (scene_target_w, scene_target_h))
    
    # 創建並排比較圖
    # 上排：模板和反扭轉後的場景
    top_row = np.hstack([template_resized, warped_resized])
    
    # 下排：場景圖像（居中）
    padding_w = (top_row.shape[1] - scene_resized.shape[1]) // 2
    padding = np.zeros((scene_resized.shape[0], padding_w, 3), dtype=np.uint8)
    bottom_row = np.hstack([padding, scene_resized, padding])
    
    # 確保上下排寬度一致
    if bottom_row.shape[1] < top_row.shape[1]:
        extra_padding = np.zeros((bottom_row.shape[0], top_row.shape[1] - bottom_row.shape[1], 3), dtype=np.uint8)
        bottom_row = np.hstack([bottom_row, extra_padding])
    
    # 合併上下排
    comparison = np.vstack([top_row, bottom_row])
    
    # 添加標籤
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    
    cv2.putText(comparison, "Template", (10, 30), font, font_scale, color, thickness)
    cv2.putText(comparison, "Warped Scene", (target_w + 10, 30), font, font_scale, color, thickness)
    cv2.putText(comparison, "Scene with Detection", (10, target_h + 30), font, font_scale, color, thickness)
    
    # 添加參數信息
    params = result['params']
    info_text = f"tx={params['tx']:.1f}, ty={params['ty']:.1f}, θ={params['theta_deg']:.1f}°, s={params['scale']:.3f}"
    cv2.putText(comparison, info_text, (10, comparison.shape[0] - 10), font, font_scale, color, thickness)
    
    return comparison


def _finalize_overlay_alpha(overlay: np.ndarray) -> None:
    if overlay.shape[2] != 4:
        raise ValueError("overlay must be RGBA")
    alpha_mask = np.any(overlay[..., :3] != 0, axis=2)
    overlay[..., 3] = 0
    overlay[..., 3][alpha_mask] = 255


def create_matches_overlay(template: np.ndarray,
                           image: np.ndarray,
                           result: Dict,
                           show_all: bool = False) -> np.ndarray:
    """建立匹配線段的透明疊圖（模板 + 場景拼接寬度）。"""
    kp1 = result['keypoints_template']
    kp2 = result['keypoints_image']
    matches = result['matches']
    inliers_mask = result['inliers_mask']

    if template.ndim == 2:
        template_color = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    else:
        template_color = template

    if image.ndim == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image

    h1, w1 = template_color.shape[:2]
    h2, w2 = image_color.shape[:2]
    vis_h = max(h1, h2)
    vis_w = w1 + w2

    overlay = np.zeros((vis_h, vis_w, 4), dtype=np.uint8)
    canvas = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)

    kp2_shifted = []
    for kp in kp2:
        kp2_shifted.append(cv2.KeyPoint(kp.pt[0] + w1, kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id))

    for i, match in enumerate(matches):
        if not show_all and not inliers_mask[i]:
            continue
        pt1 = tuple(map(int, kp1[match.queryIdx].pt))
        pt2 = tuple(map(int, kp2_shifted[match.trainIdx].pt))

        if inliers_mask[i]:
            color = (0, 255, 0)
            thickness = 2
        else:
            color = (0, 0, 255)
            thickness = 1

        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, 3, color, -1, cv2.LINE_AA)

    overlay[..., :3] = canvas
    _finalize_overlay_alpha(overlay)
    return overlay


def create_reprojection_overlay(template: np.ndarray,
                                image: np.ndarray,
                                result: Dict) -> np.ndarray:
    """建立重投影框的透明疊圖（與場景圖像同尺寸）。"""
    if image.ndim == 2:
        h, w = image.shape
    else:
        h, w = image.shape[:2]

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    M = result['affine_matrix']
    if M is not None:
        tpl_h, tpl_w = template.shape[:2]
        corners_template = np.float32([
            [0, 0],
            [tpl_w, 0],
            [tpl_w, tpl_h],
            [0, tpl_h]
        ]).reshape(-1, 1, 2)

        corners_projected = cv2.transform(corners_template, M)
        corners_projected = corners_projected.reshape(-1, 2).astype(np.int32)

        cv2.polylines(canvas, [corners_projected], True, (255, 0, 0), 2, cv2.LINE_AA)

        for idx, corner in enumerate(corners_projected):
            cv2.circle(canvas, tuple(corner), 4, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(canvas, f"P{idx+1}", (corner[0] + 5, corner[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    overlay[..., :3] = canvas
    _finalize_overlay_alpha(overlay)
    return overlay

