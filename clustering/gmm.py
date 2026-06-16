"""
LIBERO_OBJECT 数据分析脚本 — Step 7+8 统一入口
Step 7: GMM 峰数判断 (BIC 准则)  →  gmm/step7_gmm.py
Step 8: 统计汇总                →  gmm/step8_statistics.py

这是核心分析步骤:
    Step 7: 对每个采样点的 K 个 PCA action (3D) 拟合 GMM (K=1~5),
            用 BIC 准则选出最佳峰数 K*
    Step 8: 统计 all_best_k 分布，输出 Counter

用法:
    python gmm.py

输出 (保存在 gmm/ 目录下):
    gmm/all_best_k.npy         — (K,) 每个采样点的最佳 K*
    gmm/bic_curves.npy          — (K, 5) BIC 曲线
    gmm/kstar_counter.pkl       — Counter 统计分布
    gmm/kstar_summary.json      — 统计摘要 (人类可读)
"""

import os
import numpy as np

from gmm.step7_gmm import gmm_select_peaks, save_gmm_results
from gmm.step8_statistics import final_statistics, save_statistics

# ============================================================
# 配置
# ============================================================
LOCAL_ACTIONS_PCA_PATH = os.path.join(os.path.dirname(__file__), "pca_action", "local_actions_pca.npy")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "gmm")
K_RANGE = (1, 6)  # K = 1, 2, 3, 4, 5
RANDOM_SEED = 0


def main():
    output_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("LIBERO_OBJECT: Step 7 (GMM 峰数判断) + Step 8 (统计)")
    print("=" * 60)

    # 加载 local_actions_pca
    pca_path = os.path.abspath(LOCAL_ACTIONS_PCA_PATH)
    print(f"加载 local_actions_pca → {pca_path}")
    local_actions_pca = np.load(pca_path)
    local_actions_pca_list = list(local_actions_pca)

    # Step 7: GMM 峰数判断
    all_best_k, bic_curves = gmm_select_peaks(
        local_actions_pca_list, k_range=K_RANGE, random_seed=RANDOM_SEED
    )
    save_gmm_results(all_best_k, bic_curves, None, output_dir)

    # Step 8: 统计汇总
    counter, summary = final_statistics(all_best_k)
    save_statistics(counter, summary, output_dir)

    print(f"\n{'='*60}")
    print(f"保存文件 → {output_dir}/")
    print(f"  all_best_k.npy         → {all_best_k.shape}")
    print(f"  bic_curves.npy          → ({len(all_best_k)}, 5)")
    print(f"  kstar_counter.pkl       → Counter")
    print(f"  kstar_summary.json      → 统计摘要")
    print(f"{'='*60}")
    print("✅ Step 7 + Step 8 完成!")


if __name__ == "__main__":
    main()
