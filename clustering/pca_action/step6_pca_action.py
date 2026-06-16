"""
Step 6: PCA 压缩 Action

对每个采样状态的 K 个邻域 action 做 PCA(data_dim=7 → 3)

    输入:  local_actions_list  [K] 每个 (K, 7)
    输出:  local_actions_pca_list  [K] 每个 (K, 3)
           act_pca_models         [K] 每个 PCA 对象

用法:
    from pca_action.step6_pca_action import reduce_actions_pca
    local_actions_pca_list, act_pca_models = reduce_actions_pca(local_actions_list)
"""

import os
import pickle
import numpy as np
from sklearn.decomposition import PCA


def reduce_actions_pca(
    local_actions_list: list,
    n_components: int = 3,
    verbose: bool = True,
):
    """
    对每个采样点的 K 个邻域 action 做 PCA 降维

    Args:
        local_actions_list: list of (K, 7) arrays, 长度 K
        n_components: 目标维度, 默认 3
        verbose: 是否打印信息

    Returns:
        local_actions_pca_list: list of (K, n_components) arrays
        act_pca_models:         list of PCA 对象 (每个采样点一个)
    """
    if verbose:
        print(f"[Step 6] PCA 压缩 Action")
        print(f"[Step 6]   采样点数: {len(local_actions_list)}")
        print(f"[Step 6]   每个 local_action 形状: {local_actions_list[0].shape}")
        print(f"[Step 6]   降维: {local_actions_list[0].shape[1]}D → {n_components}D")

    local_actions_pca_list = []
    act_pca_models = []

    # 累积解释方差 (用于最终汇总)
    all_explained_var = []

    for i, local_actions in enumerate(local_actions_list):
        act_pca = PCA(n_components=n_components, random_state=0)
        local_actions_pca = act_pca.fit_transform(local_actions)   # (K, 3)

        local_actions_pca_list.append(local_actions_pca)
        act_pca_models.append(act_pca)
        all_explained_var.append(act_pca.explained_variance_ratio_)

        if verbose and (i + 1) % 20 == 0:
            print(f"[Step 6]   {i + 1}/{len(local_actions_list)} 完成")

    # 汇总
    mean_var = np.mean(all_explained_var, axis=0)
    cum_var = np.cumsum(mean_var)

    if verbose:
        print(f"[Step 6]   平均各主成分解释方差: {np.round(mean_var, 4)}")
        print(f"[Step 6]   平均累积解释方差: {cum_var[-1]:.4f} ({cum_var[-1]*100:.1f}%)")
        print(f"[Step 6]   local_actions_pca 形状: {local_actions_pca_list[0].shape}")
        print(f"[Step 6] Action PCA 压缩完成 ✓")

    return local_actions_pca_list, act_pca_models


def save_actions_pca(local_actions_pca_list, act_pca_models, output_dir: str, verbose: bool = True):
    """保存 PCA 压缩后的 action 和 PCA 模型"""
    # 堆叠为 (K, n_neighbors, 3)
    stacked = np.stack(local_actions_pca_list, axis=0)
    path_pca = os.path.join(output_dir, "local_actions_pca.npy")
    np.save(path_pca, stacked)

    path_models = os.path.join(output_dir, "act_pca_models.pkl")
    with open(path_models, "wb") as f:
        pickle.dump(act_pca_models, f)

    if verbose:
        print(f"[Step 6] 保存 → {path_pca}    ({stacked.shape})")
        print(f"[Step 6] 保存 → {path_models}  ({len(act_pca_models)} PCA 模型)")

    return stacked


def load_actions_pca(output_dir: str):
    """加载 PCA 压缩后的 action 和模型"""
    stacked = np.load(os.path.join(output_dir, "local_actions_pca.npy"))
    with open(os.path.join(output_dir, "act_pca_models.pkl"), "rb") as f:
        models = pickle.load(f)
    return list(stacked), models


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    OUTPUT_DIR = os.path.dirname(__file__)

    # 加载 local_actions
    local_actions = np.load(os.path.join(os.path.dirname(__file__), "..", "pca_action", "local_actions.npy"))
    local_actions_list = list(local_actions)

    pca_list, models = reduce_actions_pca(local_actions_list)
    save_actions_pca(pca_list, models, OUTPUT_DIR)
