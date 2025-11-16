#!/usr/bin/env python3
"""
SharpMatch 姿態評估腳本：
- 讀取 GT、檢測結果、模板資訊與閾值設定
- 以 (image_id, template_id) 為單位進行 TP/FP/FN 計算
- 匯出整體 Precision/Recall/F1 與逐影像統計
"""

import csv
import math
import yaml
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


@dataclass
class MatchCounters:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, tp: int, fp: int, fn: int) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn


@dataclass
class EvalResult:
    image_id: str
    template_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = field(default_factory=list)


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_template_info(path: Path) -> Dict[str, float]:
    diag = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            diag[row["template_id"]] = float(row["diag_px"])
    return diag


def load_gt(path: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    gt_by_img_tpl = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("valid", "1") == "0":
                continue
            image_id = row["image_id"]
            template_id = row["template_id"]
            gt_by_img_tpl[(image_id, template_id)].append(
                {
                    "cx": float(row["cx"]),
                    "cy": float(row["cy"]),
                    "theta": float(row["theta_deg"]),
                    "scale_gt": float(row["scale_gt"]),
                    "matched": False,
                }
            )
    return gt_by_img_tpl


def load_det(
    path: Path,
    score_thr: float,
    max_per_image: int | None,
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    det_by_img_tpl = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row["score"])
            if score < score_thr:
                continue
            key = (row["image_id"], row["template_id"])
            det_by_img_tpl[key].append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "theta": float(row["theta_deg"]),
                    "scale": float(row["scale"]),
                    "score": score,
                }
            )

    for key, dets in det_by_img_tpl.items():
        dets.sort(key=lambda d: d["score"], reverse=True)
        if max_per_image is not None and len(dets) > max_per_image:
            det_by_img_tpl[key] = dets[:max_per_image]
    return det_by_img_tpl


def angle_diff_deg(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    print(f"angle_diff_deg: {a} - {b} = {diff}")
    return abs(diff)


def is_match(det: Dict[str, Any], gt: Dict[str, Any], diag_px: float, cfg: Dict[str, Any]) -> bool:
    dx = det["x"] - gt["cx"]
    dy = det["y"] - gt["cy"]
    print(f"det['x']: {det['x']}, gt['cx']: {gt['cx']}, dx: {dx}, det['y']: {det['y']}, gt['cy']: {gt['cy']}, dy: {dy}")
    dist = math.hypot(dx, dy)
    dist_thr = cfg["distance_ratio"] * diag_px * gt["scale_gt"]

    dtheta = angle_diff_deg(det["theta"], gt["theta"])
    dscale = abs(det["scale"] - gt["scale_gt"]) / max(gt["scale_gt"], 1e-9)

    return (
        dist <= dist_thr
        and dtheta <= cfg["angle_deg"]
        and dscale <= cfg["scale_ratio"]
    )


def evaluate(
    gt: Dict[Tuple[str, str], List[Dict[str, Any]]],
    det: Dict[Tuple[str, str], List[Dict[str, Any]]],
    template_diag: Dict[str, float],
    cfg: Dict[str, Any],
) -> Tuple[MatchCounters, List[EvalResult]]:
    counters = MatchCounters()
    per_key_results: List[EvalResult] = []

    all_keys = set(gt.keys()) | set(det.keys())

    for key in sorted(all_keys):
        image_id, template_id = key
        gt_list = [dict(g) for g in gt.get(key, [])]  # copy to reset matched flags
        det_list = det.get(key, [])
        diag_px = template_diag.get(template_id, 1.0)

        result = EvalResult(image_id=image_id, template_id=template_id)

        for d in det_list:
            matched_gt = None
            for g in gt_list:
                if g.get("matched"):
                    continue
                if is_match(d, g, diag_px, cfg):
                    g["matched"] = True
                    matched_gt = g
                    break

            if matched_gt:
                counters.tp += 1
                result.tp += 1
                result.matched_pairs.append((d, matched_gt))
            else:
                counters.fp += 1
                result.fp += 1

        for g in gt_list:
            if not g.get("matched"):
                counters.fn += 1
                result.fn += 1

        per_key_results.append(result)

    return counters, per_key_results


def write_per_image_csv(results: List[EvalResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "template_id", "tp", "fp", "fn"])
        for res in results:
            writer.writerow([res.image_id, res.template_id, res.tp, res.fp, res.fn])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SharpMatch 姿態評估腳本")
    default_root = Path(__file__).resolve().parent.parent / "verify_data"
    default_cfg = Path(__file__).resolve().parent.parent / "configs" / "pose_eval.yaml"

    parser.add_argument("--data-root", type=Path, default=default_root, help="驗證資料根目錄")
    parser.add_argument("--config", type=Path, default=default_cfg, help="姿態評估設定檔路徑")
    parser.add_argument("--gt-csv", type=Path, default=None, help="GT CSV 路徑（預設 data-root/gt/gt_pose.csv）")
    parser.add_argument("--det-csv", type=Path, default=None, help="檢測 CSV 路徑（預設 data-root/det/det_sharpmatch_v1.csv）")
    parser.add_argument(
        "--template-info",
        type=Path,
        default=None,
        help="模板 meta CSV 路徑（預設 data-root/meta/template_info.csv）",
    )
    parser.add_argument(
        "--per-image-csv",
        type=Path,
        default=None,
        help="輸出 per-image 統計 CSV（預設 data-root/eval_result_per_image.csv）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config)

    data_root = args.data_root
    gt_path = args.gt_csv or (data_root / "gt" / "gt_pose.csv")
    det_path = args.det_csv or (data_root / "det" / "det_sharpmatch_v1.csv")
    tmpl_meta_path = args.template_info or (data_root / "meta" / "template_info.csv")
    per_image_output = args.per_image_csv or (data_root / "eval_result_per_image.csv")

    template_diag = load_template_info(tmpl_meta_path)
    gt = load_gt(gt_path)
    det = load_det(det_path, cfg.get("score_threshold", 0.0), cfg.get("max_dets_per_image"))

    counters, per_image_results = evaluate(gt, det, template_diag, cfg)

    precision = safe_div(counters.tp, counters.tp + counters.fp)
    recall = safe_div(counters.tp, counters.tp + counters.fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)

    print(f"TP={counters.tp}, FP={counters.fp}, FN={counters.fn}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    write_per_image_csv(per_image_results, per_image_output)
    print(f"[OK] per-image 評估結果已輸出：{per_image_output}")


if __name__ == "__main__":
    main()

