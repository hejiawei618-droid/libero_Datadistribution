"""
LIBERO_OBJECT 数据分析脚本 — Step 3 统一入口
Step 3: 建立 kNN 索引  →  knn/step3_knn.py

用法:
    python knn.py

输出 (保存在 knn/ 目录下):
    knn/knn_index.pkl   — sklearn NearestNeighbors 对象 (n_neighbors=300, euclidean)
"""

import os
import numpy as np

from knn.step3_knn import build_knn_index, save_knn

# ============================================================
# 配置
# ============================================================
STATES_PCA_PATH = os.path.join(os.path.dirname(__file__), "pca", "states_pca.npy")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "knn")
N_NEIGHBORS = 300


def main():
    output_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("LIBERO_OBJECT: Step 3 (建立 kNN 索引)")
    print("=" * 60)

    # 加载 states_pca
    states_pca_path = os.path.abspath(STATES_PCA_PATH)
    print(f"[Step 3] 加载 states_pca → {states_pca_path}")
    states_pca = np.load(states_pca_path)

    # Step 3: 建立 kNN 索引
    knn = build_knn_index(states_pca, n_neighbors=N_NEIGHBORS)

    # 保存
    save_knn(knn, output_dir)

    print(f"\n{'='*60}")
    print(f"保存文件 → {output_dir}/")
    print(f"  knn_index.pkl  → NearestNeighbors(n_neighbors={N_NEIGHBORS})")
    print(f"{'='*60}")
    print("✅ Step 3 完成!")


if __name__ == "__main__":
    main()
