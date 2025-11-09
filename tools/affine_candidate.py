#!/usr/bin/env python3
"""
仿射匹配 CLI 工具：使用特徵檢測 + RANSAC 估計仿射變換
"""

import sys
import os
import argparse
import json
import yaml
import csv
import time
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from sharp.affine_match import (
    affine_match,
    visualize_matches,
    visualize_reprojection
)


def load_config(config_path: str) -> dict:
    """從 YAML 文件載入配置。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config or {}


def merge_configs(cli_args: argparse.Namespace, yaml_config: dict) -> dict:
    """合併 CLI 參數和 YAML 配置（CLI 優先）。"""
    config = {}
    
    # 從 YAML 載入
    if yaml_config:
        config.update(yaml_config)
    
    # CLI 參數覆蓋 YAML
    if cli_args.template:
        config['template'] = cli_args.template
    if cli_args.image:
        config['image'] = cli_args.image
    if cli_args.detector:
        config['detector'] = cli_args.detector
    if cli_args.ratio is not None:
        config['ratio'] = cli_args.ratio
    if cli_args.max_kp is not None:
        config['max_keypoints'] = cli_args.max_kp
    if cli_args.ransac_reproj_thresh is not None:
        config['ransac_reproj_thresh'] = cli_args.ransac_reproj_thresh
    if cli_args.ransac_max_iter is not None:
        config['ransac_max_iters'] = cli_args.ransac_max_iter
    if cli_args.ransac_conf is not None:
        config['ransac_confidence'] = cli_args.ransac_conf
    if cli_args.min_inliers is not None:
        config['min_inliers'] = cli_args.min_inliers
    if cli_args.min_inlier_ratio is not None:
        config['min_inlier_ratio'] = cli_args.min_inlier_ratio
    if cli_args.outdir:
        config['outdir'] = cli_args.outdir
    if cli_args.debug:
        config['debug'] = True
    
    return config


def save_results(result: dict, output_dir: str, config: dict, 
                 template_path: str, image_path: str, elapsed_time_ms: float):
    """保存所有輸出結果。
    
    Args:
        result: 匹配結果字典
        output_dir: 輸出目錄
        config: 配置字典
        template_path: 模板圖像路徑
        image_path: 場景圖像路徑
        elapsed_time_ms: 匹配耗時（毫秒）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存 JSON 結果
    json_data = {
        'affine_matrix': result['affine_matrix'].tolist() if result['affine_matrix'] is not None else None,
        'params': result['params'],
        'statistics': {
            'num_matches': result['num_matches'],
            'num_inliers': result['num_inliers'],
            'num_outliers': result['num_outliers'],
            'inlier_ratio': result['inlier_ratio']
        },
        'config': {
            'detector_type': result['detector_type'],
            'ratio_threshold': result['ratio_threshold'],
            'ransac_params': result['ransac_params']
        }
    }
    
    json_path = output_path / 'affine.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON 已保存: {json_path}")
    
    # 2. 保存內點遮罩
    inliers_mask = result['inliers_mask']
    npy_path = output_path / 'inliers_mask.npy'
    np.save(npy_path, inliers_mask)
    print(f"  ✓ 內點遮罩已保存: {npy_path}")
    
    # 3. 保存 CSV 摘要
    csv_path = output_path / 'summary.csv'
    file_exists = csv_path.exists()
    
    # 提取文件名（不含路徑）
    template_filename = Path(template_path).name
    image_filename = Path(image_path).name
    filename = f"{template_filename} -> {image_filename}"
    
    # 準備 CSV 行數據
    params = result['params']
    csv_row = {
        '檔名': filename,
        '特徵器': result['detector_type'],
        '總匹配': result['num_matches'],  # 通過 ratio 的匹配數
        '通過 ratio 的匹配': result['num_matches'],
        'inliers': result['num_inliers'],
        'inlier_ratio': f"{result['inlier_ratio']:.4f}",
        'theta_deg': f"{params['theta_deg']:.2f}",
        'scale': f"{params['scale']:.4f}",
        'tx': f"{params['tx']:.2f}",
        'ty': f"{params['ty']:.2f}",
        '耗時(ms)': f"{elapsed_time_ms:.2f}"
    }
    
    # 定義 CSV 欄位順序
    fieldnames = ['檔名', '特徵器', '總匹配', '通過 ratio 的匹配', 'inliers', 
                  'inlier_ratio', 'theta_deg', 'scale', 'tx', 'ty', '耗時(ms)']
    
    # 寫入 CSV（追加模式）
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:  # utf-8-sig 用於 Excel 正確顯示中文
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # 如果是新文件，寫入表頭
        writer.writerow(csv_row)
    
    print(f"  ✓ CSV 摘要已保存: {csv_path}")


def run_affine_matching(template_path: str, image_path: str,
                        detector_type: str = 'auto',
                        ratio: float = 0.75,
                        max_keypoints: int = 3000,
                        ransac_reproj_thresh: float = 3.0,
                        ransac_max_iters: int = 2000,
                        ransac_confidence: float = 0.99,
                        min_inliers: int = 30,
                        min_inlier_ratio: float = 0.25,
                        output_dir: str = 'outputs',
                        debug: bool = False) -> int:
    """核心匹配邏輯（可被 import 調用）。
    
    Args:
        template_path: 模板圖像路徑
        image_path: 場景圖像路徑
        detector_type: 檢測器類型
        ratio: Lowe Ratio Test 閾值
        max_keypoints: 最多特徵點數量
        ransac_reproj_thresh: RANSAC 重投影閾值
        ransac_max_iters: RANSAC 最大迭代次數
        ransac_confidence: RANSAC 置信度
        min_inliers: 最小內點數量門檻
        min_inlier_ratio: 最小內點比率門檻
        output_dir: 輸出目錄
        debug: 是否顯示詳細統計信息
    
    Returns:
        0 表示成功，1 表示一般錯誤，2 表示品質不達標
    """
    # 載入圖像
    print("=" * 60)
    print("仿射匹配工具")
    print("=" * 60)
    print(f"\n載入圖像...")
    print(f"  模板: {template_path}")
    print(f"  場景: {image_path}")
    
    template = cv2.imread(template_path)
    if template is None:
        print(f"錯誤: 無法載入模板圖像: {template_path}")
        return 1
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"錯誤: 無法載入場景圖像: {image_path}")
        return 1
    
    print(f"  模板尺寸: {template.shape[1]}x{template.shape[0]}")
    print(f"  場景尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 執行匹配（計時）
    print(f"\n執行匹配...")
    print(f"  檢測器: {detector_type}")
    print(f"  Ratio 閾值: {ratio}")
    print(f"  最多特徵點: {max_keypoints}")
    print(f"  RANSAC 參數: reproj_thresh={ransac_reproj_thresh}, max_iter={ransac_max_iters}, confidence={ransac_confidence}")
    
    start_time = time.time()
    try:
        result = affine_match(
            template, image,
            detector_type=detector_type,
            ratio_threshold=ratio,
            max_keypoints=max_keypoints,
            ransac_reproj_thresh=ransac_reproj_thresh,
            ransac_max_iters=ransac_max_iters,
            ransac_confidence=ransac_confidence
        )
    except Exception as e:
        print(f"\n錯誤: {e}")
        return 1
    elapsed_time_ms = (time.time() - start_time) * 1000  # 轉換為毫秒
    
    # 輸出結果
    print(f"\n匹配成功！")
    print(f"  檢測器類型: {result['detector_type']}")
    print(f"  匹配點數量: {result['num_matches']}")
    print(f"  內點數量: {result['num_inliers']}")
    print(f"  外點數量: {result['num_outliers']}")
    print(f"  內點比率: {result['inlier_ratio']:.2%}")
    
    # 品質門檻檢查
    num_inliers = result['num_inliers']
    inlier_ratio = result['inlier_ratio']
    quality_failed = False
    quality_issues = []
    
    if num_inliers < min_inliers:
        quality_failed = True
        quality_issues.append(f"內點數量不足: {num_inliers} < {min_inliers}")
    
    if inlier_ratio < min_inlier_ratio:
        quality_failed = True
        quality_issues.append(f"內點比率不足: {inlier_ratio:.2%} < {min_inlier_ratio:.2%}")
    
    if quality_failed:
        print(f"\n{'=' * 60}")
        print("品質檢查失敗")
        print("=" * 60)
        print(f"\n當前結果:")
        print(f"  內點數量: {num_inliers} (要求 >= {min_inliers})")
        print(f"  內點比率: {inlier_ratio:.2%} (要求 >= {min_inlier_ratio:.2%})")
        print(f"\n問題:")
        for issue in quality_issues:
            print(f"  ✗ {issue}")
        print(f"\n建議:")
        print(f"  1. 提高特徵點數量: --max-kp 5000 (當前: {max_keypoints})")
        print(f"  2. 放寬 ratio 閾值: --ratio 0.7 (當前: {ratio})")
        print(f"  3. 放寬 RANSAC 重投影閾值: --ransac-reproj-thresh 5.0 (當前: {ransac_reproj_thresh})")
        print("=" * 60)
        return 2  # 品質不達標
    
    # 解析參數
    params = result['params']
    print(f"\n仿射變換參數:")
    print(f"  平移 (tx, ty): ({params['tx']:.2f}, {params['ty']:.2f})")
    print(f"  旋轉角度: {params['theta_deg']:.2f}° ({params['theta_rad']:.4f} rad)")
    print(f"  縮放因子: {params['scale']:.4f}")
    
    if debug:
        print(f"\n詳細統計:")
        print(f"  縮放因子 (x): {params['scale_x']:.4f}")
        print(f"  縮放因子 (y): {params['scale_y']:.4f}")
        print(f"  比例差異: {abs(params['scale_x'] - params['scale_y']):.4f}")
    
    # 生成可視化
    print(f"\n生成可視化結果...")
    
    # 1. 匹配可視化（所有匹配點）
    matches_vis = visualize_matches(template, image, result, show_all=True)
    matches_path = Path(output_dir) / 'matches_vis.png'
    cv2.imwrite(str(matches_path), matches_vis)
    print(f"  ✓ 匹配可視化已保存: {matches_path}")
    
    # 2. 只顯示內點的匹配可視化
    matches_inliers_vis = visualize_matches(template, image, result, show_all=False)
    matches_inliers_path = Path(output_dir) / 'matches_inliers_only.png'
    cv2.imwrite(str(matches_inliers_path), matches_inliers_vis)
    print(f"  ✓ 內點匹配可視化已保存: {matches_inliers_path}")
    
    # 3. 重投影可視化
    reproj_vis = visualize_reprojection(template, image, result)
    reproj_path = Path(output_dir) / 'reproject.png'
    cv2.imwrite(str(reproj_path), reproj_vis)
    print(f"  ✓ 重投影可視化已保存: {reproj_path}")
    
    # 保存結果
    config = {
        'detector': detector_type,
        'ratio': ratio,
        'max_keypoints': max_keypoints,
        'ransac_reproj_thresh': ransac_reproj_thresh,
        'ransac_max_iters': ransac_max_iters,
        'ransac_confidence': ransac_confidence
    }
    save_results(result, output_dir, config, template_path, image_path, elapsed_time_ms)
    
    print(f"\n所有結果已保存到: {output_dir}/")
    print("=" * 60)
    
    return 0


def main(args=None):
    """CLI 入口點：解析參數並調用核心邏輯。
    
    Args:
        args: 可選的命令行參數列表（None 表示使用 sys.argv）
    
    Returns:
        0 表示成功，1 表示失敗
    """
    parser = argparse.ArgumentParser(
        description='使用特徵檢測 + RANSAC 估計仿射變換',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --template data/demo/template.png --image data/demo/scene.png
  %(prog)s --template tpl.png --image img.png --detector sift --ratio 0.7
  %(prog)s --config configs/affine.yaml --template tpl.png --image img.png
        """
    )
    
    # 必需參數
    parser.add_argument('--template', type=str, help='模板圖像路徑')
    parser.add_argument('--image', type=str, help='場景圖像路徑')
    
    # 檢測器選項
    parser.add_argument('--detector', type=str, default='auto',
                       choices=['auto', 'sift', 'orb', 'akaze'],
                       help='特徵檢測器類型 (預設: auto)')
    
    # 匹配參數
    parser.add_argument('--ratio', type=float, default=0.75,
                       help='Lowe Ratio Test 閾值 (預設: 0.75)')
    parser.add_argument('--max-kp', type=int, default=3000,
                       help='最多特徵點數量 (預設: 3000)')
    
    # RANSAC 參數
    parser.add_argument('--ransac-reproj-thresh', type=float, default=3.0,
                       help='RANSAC 重投影閾值（像素）(預設: 3.0)')
    parser.add_argument('--ransac-max-iter', type=int, default=2000,
                       help='RANSAC 最大迭代次數 (預設: 2000)')
    parser.add_argument('--ransac-conf', type=float, default=0.99,
                       help='RANSAC 置信度 (預設: 0.99)')
    
    # 品質門檻參數
    parser.add_argument('--min-inliers', type=int, default=30,
                       help='最小內點數量門檻 (預設: 30)')
    parser.add_argument('--min-inlier-ratio', type=float, default=0.25,
                       help='最小內點比率門檻 (預設: 0.25)')
    
    # 輸出選項
    parser.add_argument('--outdir', type=str, default='outputs',
                       help='輸出目錄 (預設: outputs)')
    parser.add_argument('--config', type=str, help='YAML 配置文件路徑')
    parser.add_argument('--debug', action='store_true',
                       help='顯示詳細統計信息')
    
    args = parser.parse_args(args)
    
    # 載入配置
    yaml_config = {}
    if args.config:
        if not os.path.exists(args.config):
            print(f"錯誤: 配置文件不存在: {args.config}")
            return 1
        yaml_config = load_config(args.config)
    
    # 合併配置
    config = merge_configs(args, yaml_config)
    
    # 檢查必需參數
    if 'template' not in config or 'image' not in config:
        parser.error("必須提供 --template 和 --image 參數（或通過配置文件）")
    
    # 提取參數
    template_path = config['template']
    image_path = config['image']
    detector_type = config.get('detector', 'auto')
    ratio = config.get('ratio', 0.75)
    max_keypoints = config.get('max_keypoints', 3000)
    ransac_reproj_thresh = config.get('ransac_reproj_thresh', 3.0)
    # 支援兩種參數名稱以保持向後兼容
    ransac_max_iters = config.get('ransac_max_iters', config.get('ransac_max_iter', 2000))
    ransac_confidence = config.get('ransac_confidence', 0.99)
    min_inliers = config.get('min_inliers', 30)
    min_inlier_ratio = config.get('min_inlier_ratio', 0.25)
    output_dir = config.get('outdir', 'outputs')
    debug = config.get('debug', False)
    
    # 調用核心邏輯
    return run_affine_matching(
        template_path=template_path,
        image_path=image_path,
        detector_type=detector_type,
        ratio=ratio,
        max_keypoints=max_keypoints,
        ransac_reproj_thresh=ransac_reproj_thresh,
        ransac_max_iters=ransac_max_iters,
        ransac_confidence=ransac_confidence,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
        output_dir=output_dir,
        debug=debug
    )


if __name__ == '__main__':
    sys.exit(main())

