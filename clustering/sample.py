"""
LIBERO_OBJECT 数据分析脚本 — Step 4 统一入口
Step 4: 随机采样 500 个中心状态  →  sample/step4_sample.py

用法:
    python sample.py

输出 (保存在 sample/ 目录下):
    sample/sample_ids.npy   — (500,) 随机选中的状态索引
"""

import os
import numpy as np

from sample.step4_sample import random_sample_states, save_sample_ids

# ============================================================
# 配置
# ============================================================
STATES_PCA_PATH = os.path.join(os.path.dirname(__file__), "pca", "states_pca.npy")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample")
N_SAMPLES = 500  # 可在 run_all.py 中统一修改
RANDOM_SEED = 0


def main():
    output_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"LIBERO_OBJECT: Step 4 (随机采样 {N_SAMPLES} 个中心状态)")
    print("=" * 60)

    # 加载 states_pca
    states_pca_path = os.path.abspath(STATES_PCA_PATH)
    print(f"[Step 4] 加载 states_pca → {states_pca_path}")
    states_pca = np.load(states_pca_path)

    # Step 4: 随机采样
    sample_ids = random_sample_states(states_pca, n_samples=N_SAMPLES, random_seed=RANDOM_SEED)

    # 保存
    save_sample_ids(sample_ids, output_dir)

    print(f"\n{'='*60}")
    print(f"保存文件 → {output_dir}/")
    print(f"  sample_ids.npy  → {sample_ids.shape}, 前10个: {sample_ids[:10]}")
    print(f"{'='*60}")
    print("✅ Step 4 完成!")


if __name__ == "__main__":
    main()
