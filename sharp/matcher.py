"""
特徵匹配器模組：支援 Brute Force (BF) 和 FLANN 匹配器
"""

import cv2
import numpy as np
from typing import Union


def create_matcher(detector_type: str, matcher_type: str = 'bf') -> Union[cv2.BFMatcher, cv2.FlannBasedMatcher]:
    """創建特徵匹配器。
    
    Args:
        detector_type: 檢測器類型 ('sift', 'orb', 'akaze')
        matcher_type: 匹配器類型 ('bf' 或 'flann')
    
    Returns:
        匹配器物件 (BFMatcher 或 FlannBasedMatcher)
    
    Raises:
        ValueError: 如果 matcher_type 或 detector_type 不支援
    """
    matcher_type = matcher_type.lower()
    detector_type = detector_type.lower()
    
    if matcher_type == 'bf':
        # Brute Force 匹配器
        if detector_type in ['sift', 'akaze']:
            # SIFT 和 AKAZE 使用 L2 距離
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif detector_type == 'orb':
            # ORB 使用 Hamming 距離
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"不支援的檢測器類型: {detector_type}")
    
    elif matcher_type == 'flann':
        # FLANN 匹配器
        if detector_type in ['sift', 'akaze']:
            # SIFT 和 AKAZE 使用 KDTree 索引（浮點描述子）
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # 搜索次數，越多越準確但越慢
            return cv2.FlannBasedMatcher(index_params, search_params)
        
        elif detector_type == 'orb':
            # ORB 使用 LSH 索引（二進制描述子）
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,      # LSH 哈希表數量
                key_size=12,         # 鍵大小
                multi_probe_level=1  # 多探測級別
            )
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        
        else:
            raise ValueError(f"不支援的檢測器類型: {detector_type}")
    
    else:
        raise ValueError(f"不支援的匹配器類型: {matcher_type}。支援: 'bf', 'flann'")


def match_features(desc1: np.ndarray, desc2: np.ndarray,
                   detector_type: str,
                   matcher_type: str = 'bf',
                   ratio_threshold: float = 0.75) -> list:
    """使用 KNN 匹配（k=2）和 Lowe Ratio Test 進行特徵匹配。
    
    Args:
        desc1: 模板描述子
        desc2: 場景描述子
        detector_type: 檢測器類型（用於選擇距離度量）
        matcher_type: 匹配器類型 ('bf' 或 'flann')
        ratio_threshold: Lowe Ratio Test 閾值
    
    Returns:
        匹配列表 (List[cv2.DMatch])
    
    Raises:
        ValueError: 如果匹配失敗
    """
    # 創建匹配器
    matcher = create_matcher(detector_type, matcher_type)
    
    # 確保描述子類型正確（FLANN 需要 float32）
    if matcher_type == 'flann' and detector_type in ['sift', 'akaze']:
        if desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
        if desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32)
    
    try:
        # KNN 匹配（k=2）
        knn_matches = matcher.knnMatch(desc1, desc2, k=2)
    except cv2.error as e:
        raise ValueError(f"匹配失敗: {e}。建議：1) 檢查描述子格式 2) 確保描述子不為空")
    
    # Lowe Ratio Test
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches


