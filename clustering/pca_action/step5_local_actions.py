"""
Step 5: 获取邻域 Action

对每个采样的中心状态，通过 kNN 找到 1000 个最近邻，提取对应的 actions

    输入:  sample_ids   (K,)
           knn           NearestNeighbors
           states_pca    (N, 10)
           actions       (N, 7)
    输出:  local_actions_list  [K] 每个 (1000, 7)

用法:
    from pca_action.step5_local_actions import get_local_actions_for_samples
    local_actions_list = get_local_actions_for_samples(sample_ids, knn, states_pca, actions)
"""

import os
import pickle
import numpy as np


def get_local_actions_for_samples(
    sample_ids: np.ndarray,
    knn,
    states_pca: np.ndarray,
    actions: np.ndarray,
    n_neighbors: int = 1000,
    verbose: bool = True,
):
    """
    对每个采样状态，查询 kNN 获取 1000 个邻域 action

    Args:
        sample_ids:  (K,) 采样状态索引, 例如 (500,)
        knn:         sklearn NearestNeighbors 对象
        states_pca:  (N, D) PCA state 特征
        actions:     (N, Da) 原始 action
        n_neighbors: kNN 邻域大小, 默认 1000
        verbose:     是否打印信息

    Returns:
        local_actions_list: list of (n_neighbors, Da) arrays, 长度 K
        all_neighbor_ids:   list of (n_neighbors,) arrays, kNN 返回的邻域索引
    """
    if verbose:
        print(f"[Step 5] 获取邻域 Action")
        print(f"[Step 5]   采样数量: {len(sample_ids)}")
        print(f"[Step 5]   kNN k = {n_neighbors}")

    local_actions_list = []
    all_neighbor_ids = []

    for idx, center_id in enumerate(sample_ids):
        center_vec = states_pca[center_id].reshape(1, -1)  # (1, D)

        # kNN 查询
        dist, neighbor_idx = knn.kneighbors(center_vec)     # dist: (1, K), idx: (1, K)
        neighbor_idx = neighbor_idx[0]                       # (K,)

        # 提取对应 action
        local_actions = actions[neighbor_idx]                # (K, Da)
        local_actions_list.append(local_actions)
        all_neighbor_ids.append(neighbor_idx)

        if verbose and (idx + 1) % 20 == 0:
            print(f"[Step 5]   {idx + 1}/{len(sample_ids)} 完成")

    if verbose:
        print(f"[Step 5]   每个 local_actions 形状: {local_actions_list[0].shape}")
        print(f"[Step 5] 邻域 action 提取完成 ✓")

    return local_actions_list, all_neighbor_ids


def save_local_actions(local_actions_list, all_neighbor_ids, output_dir: str, verbose: bool = True):
    """保存所有 local_actions 到磁盘"""
    # 堆叠为 (K, n_neighbors, Da)
    stacked = np.stack(local_actions_list, axis=0)          # (K, n_neighbors, 7)
    stacked_idx = np.stack(all_neighbor_ids, axis=0)         # (K, n_neighbors)

    path_acts = os.path.join(output_dir, "local_actions.npy")
    path_idx = os.path.join(output_dir, "neighbor_ids.npy")
    np.save(path_acts, stacked)
    np.save(path_idx, stacked_idx)

    if verbose:
        print(f"[Step 5] 保存 → {path_acts}  ({stacked.shape})")
        print(f"[Step 5] 保存 → {path_idx}  ({stacked_idx.shape})")

    return stacked, stacked_idx


def load_local_actions(output_dir: str):
    """加载 local_actions"""
    stacked = np.load(os.path.join(output_dir, "local_actions.npy"))
    stacked_idx = np.load(os.path.join(output_dir, "neighbor_ids.npy"))
    return list(stacked), list(stacked_idx)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    OUTPUT_DIR = os.path.dirname(__file__)

    # 加载依赖数据
    sample_ids = np.load(os.path.join(os.path.dirname(__file__), "..", "sample", "sample_ids.npy"))
    states_pca = np.load(os.path.join(os.path.dirname(__file__), "..", "pca", "states_pca.npy"))
    actions = np.load(os.path.join(os.path.dirname(__file__), "..", "pca", "actions.npy"))
    with open(os.path.join(os.path.dirname(__file__), "..", "knn", "knn_index.pkl"), "rb") as f:
        knn = pickle.load(f)

    local_actions_list, neighbor_ids = get_local_actions_for_samples(
        sample_ids, knn, states_pca, actions
    )
    save_local_actions(local_actions_list, neighbor_ids, OUTPUT_DIR)
