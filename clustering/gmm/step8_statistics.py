"""
Step 8: 统计汇总 — 对 GMM 峰数判断结果做最终统计

    输入:  all_best_k  (100,) 每个采样点的最佳 K*
    输出:  counter      Counter 分布
           summary      dict

    例如: {1: 84, 2: 13, 3: 3}

用法:
    from gmm.step8_statistics import final_statistics
    counter, summary = final_statistics(all_best_k)
"""

import os
import json
import pickle
import numpy as np
from collections import Counter


def final_statistics(all_best_k: np.ndarray, verbose: bool = True):
    """
    对 all_best_k 做最终统计分析

    Args:
        all_best_k: (K_samples,) 最佳峰数
        verbose: 是否打印结果

    Returns:
        counter:  Counter 对象
        summary:  dict 包含 counts, percentages, 分类统计
    """
    K_samples = len(all_best_k)
    counter = Counter(all_best_k)

    # 分类: 单峰 / 双峰 / 三峰+
    single_peak = counter.get(1, 0)
    double_peak = counter.get(2, 0)
    multi_peak  = sum(counter.get(k, 0) for k in range(3, max(counter.keys() or [0]) + 1))

    summary = {
        "n_samples": K_samples,
        "k_range_searched": [1, 2, 3, 4, 5],
        "counts": {int(k): int(v) for k, v in sorted(counter.items())},
        "percentages": {int(k): round(v / K_samples * 100, 1)
                        for k, v in sorted(counter.items())},
        "classification": {
            "单峰 K=1":  {"count": single_peak, "percent": round(single_peak / K_samples * 100, 1)},
            "双峰 K=2":  {"count": double_peak, "percent": round(double_peak / K_samples * 100, 1)},
            "三峰+ K≥3": {"count": multi_peak,  "percent": round(multi_peak  / K_samples * 100, 1)},
        },
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Step 8 最终统计结果")
        print(f"{'='*50}")
        print(f"  样本数: {K_samples}")
        print(f"  Counter: {dict(sorted(counter.items()))}")
        print(f"\n  分类统计:")
        print(f"    单峰 (K=1):  {single_peak:>3} ({single_peak/K_samples*100:5.1f}%)")
        print(f"    双峰 (K=2):  {double_peak:>3} ({double_peak/K_samples*100:5.1f}%)")
        print(f"    三峰+(K≥3):  {multi_peak:>3} ({multi_peak/K_samples*100:5.1f}%)")
        print(f"{'='*50}")

    return counter, summary


def save_statistics(counter, summary, output_dir: str, verbose: bool = True):
    """保存统计结果 (JSON + pickle)"""
    # 保存 counter
    with open(os.path.join(output_dir, "kstar_counter.pkl"), "wb") as f:
        pickle.dump(counter, f)

    # 保存 summary JSON (人类可读)
    with open(os.path.join(output_dir, "kstar_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"[Step 8] 保存 → {output_dir}/")
        print(f"  kstar_counter.pkl   → Counter")
        print(f"  kstar_summary.json  → 统计摘要 (JSON)")


def load_statistics(output_dir: str):
    """加载统计结果"""
    with open(os.path.join(output_dir, "kstar_counter.pkl"), "rb") as f:
        counter = pickle.load(f)
    with open(os.path.join(output_dir, "kstar_summary.json"), "r") as f:
        summary = json.load(f)
    return counter, summary


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    OUTPUT_DIR = os.path.dirname(__file__)

    all_best_k = np.load(os.path.join(OUTPUT_DIR, "all_best_k.npy"))
    counter, summary = final_statistics(all_best_k)
    save_statistics(counter, summary, OUTPUT_DIR)
