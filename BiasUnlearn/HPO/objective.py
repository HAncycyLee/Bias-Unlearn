# hpo/objective.py
# 这里是要求hpo找到的优化目标，后续需要重新设计优化函数

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any

BASES = [49.0, 49.0, 49.0, 48.0]

def load_last_eval(metrics_path: str) -> Optional[Dict[str, Any]]:
    path = Path(metrics_path)
    if not path.exists():
        return None

    last_eval = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "eval":
                last_eval = row
    return last_eval

def compute_gap_stats(m: Dict[str, Any]) -> Dict[str, float]:
    gaps = {
        "gap_gender": abs(float(m["ss_gender"]) - BASES[0]),
        "gap_profession": abs(float(m["ss_profession"]) - BASES[1]),
        "gap_race": abs(float(m["ss_race"]) - BASES[2]),
        "gap_religion": abs(float(m["ss_religion"]) - BASES[3]),
    }
    gaps["mean_gap"] = sum(gaps.values()) / 4.0
    return gaps

def compute_objective(
    m: Dict[str, Any],
    gap_base: float,
    lm_base: float,
) -> float:
    gaps = compute_gap_stats(m)
    mean_gap = gaps["mean_gap"]

    gap_gain = (gap_base - mean_gap) / max(gap_base, 1e-8)
    lm_drop = lm_base - float(m["lm_overall"])
    icat_term = float(m["icat_overall"]) / 100.0

    # Day2 先用一个稳健的单目标代理分数
    score = gap_gain - 0.25 * max(0.0, lm_drop - 2.0) + 0.05 * icat_term
    return float(score)

def should_hard_prune(
    metrics: Dict[str, Any],
    lm_base: float,
    first_forget_loss: float | None = None,
) -> tuple[bool, str]:
    # 1) 数值爆炸
    for k in ["lm_overall", "icat_overall", "ss_gender", "ss_profession", "ss_race", "ss_religion"]:
        v = metrics.get(k)
        if v is None:
            continue
        if str(v).lower() in {"nan", "inf", "-inf"}:
            return True, f"bad_metric:{k}"

    # 2) utility 明显崩
    lm_now = float(metrics["lm_overall"])
    if lm_now < lm_base - 6.0:
        return True, "lm_drop_too_large"

    # 3) 这里预留给你后面接 train_step loss 的规则
    # 比如 forgetting loss 在 ~100 steps 失控
    return False, ""