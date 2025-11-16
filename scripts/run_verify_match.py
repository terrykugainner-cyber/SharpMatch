#!/usr/bin/env python3
"""
SharpMatch 檢測驗證腳本：
- 讀取 verify_data/templates 下的模板（預設 tempA.png）
- 批次處理 verify_data/images_eval 中的所有影像
- 使用 Chamfer + Affine 多物件匹配（沿用 run_demo_affine 方案 3 參數）
- 將結果輸出為 verify_data/det/det_sharpmatch_v1.csv
"""

import sys
import time
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

import cv2

# --- 專案根目錄加入匯入路徑 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sharp.affine_match import multi_affine_match_hybrid  # noqa: E402


# --- 匹配參數：沿用 run_demo_affine.py 多物件方案 ---
MATCH_CONFIG: Dict[str, Any] = dict(
    chamfer_k=20,
    chamfer_min_dist=50,
    chamfer_roi=None,
    affine_roi_padding=50,
    detector_type="auto",
    matcher_type="flann",
    ratio_threshold=0.75,
    max_keypoints=3000,
    ransac_reproj_thresh=3.0,
    ransac_max_iters=2000,
    ransac_confidence=0.99,
    min_inlier_ratio=0.35,
    min_matches=6,
    min_inliers=5,
    scale_range=(0.7, 1.3),
    min_chamfer_score=0.75,
    nms_min_distance=50.0,
    nms_min_angle_diff=5.0,
    nms_min_scale_diff=0.05,
    post_min_inlier_ratio=0.45,
    post_min_matches=8,
    post_min_inliers=6,
    post_scale_range=(0.75, 1.25),
    post_min_chamfer_score=0.90,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SharpMatch 驗證集匹配腳本")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "verify_data",
        help="驗證資料根目錄（預設：專案根目錄下 verify_data）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="輸出 CSV 路徑；未指定時寫入 data_root/det/det_sharpmatch_v1.csv",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="tempA.png",
        help="模板檔名（位於 templates/ 下）",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default="sharp_v1",
        help="CSV 欄位 method_name 的輸出內容",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="顯示 Chamfer + Affine 詳細訊息",
    )
    return parser.parse_args()


def load_image(path: Path) -> cv2.Mat:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"無法讀取影像：{path}")
    return image


def run_match_on_image(
    image_path: Path,
    template_id: str,
    template_img: cv2.Mat,
    verbose: bool,
) -> List[Dict[str, Any]]:
    image = load_image(image_path)
    start_time = time.time()
    results = multi_affine_match_hybrid(
        template_img,
        image,
        verbose=verbose,
        **MATCH_CONFIG,
    )
    elapsed_ms = (time.time() - start_time) * 1000.0

    detections: List[Dict[str, Any]] = []

    if results:
        for det_id, result in enumerate(results):
            params = result["params"]
            # 計算該實例的定位中心（模板中心投影到場景座標）
            tpl_center = np.array([[[template_img.shape[1] / 2.0, template_img.shape[0] / 2.0]]], dtype=np.float32)
            center_projected = cv2.transform(tpl_center, result["affine_matrix"])
            cx, cy = center_projected[0, 0]
            
            score = float(result.get("chamfer_score", result.get("inlier_ratio", 0.0)))
            detections.append(
                dict(
                    template_id=template_id,
                    det_id=det_id,
                    x=float(cx), #float(params["tx"]),
                    y=float(cy), #float(params["ty"]),
                    theta_deg=float(params["theta_deg"]),
                    scale=float(params["scale"]),
                    score=score,
                    runtime_ms=elapsed_ms,
                )
            )
    else:
        detections.append(
            dict(
                template_id=template_id,
                det_id=0,
                x=0.0,
                y=0.0,
                theta_deg=0.0,
                scale=0.0,
                score=0.0,
                runtime_ms=0.0,
            )
        )

    return detections


def write_csv(rows: List[List[Any]], header: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    data_root = args.data_root
    templates_dir = data_root / "templates"
    images_dir = data_root / "images_eval"
    output_path = args.output or (data_root / "det" / "det_sharpmatch_v1.csv")

    template_path = templates_dir / args.template
    template_img = load_image(template_path)
    template_id = template_path.stem

    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"找不到任何驗證影像於：{images_dir}")

    header = [
        "image_id",
        "template_id",
        "det_id",
        "x",
        "y",
        "theta_deg",
        "scale",
        "score",
        "method_name",
        "runtime_ms",
    ]

    rows: List[List[Any]] = []

    print(f"[INFO] 使用模板：{template_path.name}（{template_img.shape[1]}x{template_img.shape[0]}）")
    print(f"[INFO] 共 {len(image_paths)} 張影像待匹配")

    for idx, image_path in enumerate(image_paths, start=1):
        image_id = image_path.stem
        print(f"[INFO] ({idx}/{len(image_paths)}) 處理：{image_path.name}")

        detections = run_match_on_image(image_path, template_id, template_img, args.verbose)
        for det in detections:
            rows.append(
                [
                    image_id,
                    det["template_id"],
                    det["det_id"],
                    det["x"],
                    det["y"],
                    det["theta_deg"],
                    det["scale"],
                    det["score"],
                    args.method_name,
                    det["runtime_ms"],
                ]
            )

    write_csv(rows, header, output_path)
    print(f"[OK] 匹配結果已寫入：{output_path}")


if __name__ == "__main__":
    main()

