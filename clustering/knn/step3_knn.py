"""
Step 3: 在 PCA 降维后的 states 上建立 kNN 索引

    输入:  states_pca  (N, 10)
    输出:  knn 索引    (sklearn NearestNeighbors, n_neighbors=300, metric="euclidean")

用法:
    from knn.step3_knn import build_knn_index
    knn = build_knn_index(states_pca)
"""

import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn_index(states_pca: np.ndarray, n_neighbors: int = 1000, metric: str = "euclidean", verbose: bool = True):
    """
    在 PCA 降维后的 states 上建立 kNN 索引

    Args:
        states_pca: (N, D) PCA 降维后的 state 特征, 例如 (66451, 10)
        n_neighbors: k, 邻域大小, 默认 1000
        metric: 距离度量, 默认 "euclidean"
        verbose: 是否打印信息

    Returns:
        knn: 训练好的 NearestNeighbors 对象
    """
    if verbose:
        print(f"[Step 3] 建立 kNN 索引")
        print(f"[Step 3]   数据形状: {states_pca.shape}")
        print(f"[Step 3]   n_neighbors = {n_neighbors}")
        print(f"[Step 3]   metric = {metric}")

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(states_pca)

    if verbose:
        print(f"[Step 3] kNN 索引构建完成 ✓")

    return knn


def save_knn(knn, output_dir: str, filename: str = "knn_index.pkl"):
    """保存 kNN 索引到磁盘"""
    path = os.path.join(output_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(knn, f)
    print(f"[Step 3] 保存 → {path}")


def load_knn(output_dir: str, filename: str = "knn_index.pkl"):
    """加载 kNN 索引"""
    path = os.path.join(output_dir, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # 独立测试 Step 3: 加载 states_pca, 建 kNN, 保存
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    OUTPUT_DIR = os.path.dirname(__file__)

    states_pca_path = os.path.join(os.path.dirname(__file__), "..", "pca", "states_pca.npy")
    states_pca = np.load(states_pca_path)

    knn = build_knn_index(states_pca)
    save_knn(knn, OUTPUT_DIR)
