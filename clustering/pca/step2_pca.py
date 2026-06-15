"""
Step 2: PCA 降维 states → states_pca

用法:
    from pca.step2_pca import reduce_states_pca
    states_pca, pca = reduce_states_pca(states, n_components=10)
"""

import pickle
import numpy as np
from sklearn.decomposition import PCA


def reduce_states_pca(states: np.ndarray, n_components: int = 10, verbose: bool = True):
    """
    对 states 做 PCA 降维

    Args:
        states:        (N, Ds) 原始 state 特征
        n_components:  目标维度，默认 10
        verbose:       是否打印进度信息

    Returns:
        states_pca: (N, n_components) 降维后的 state
        pca:        训练好的 PCA 对象
    """
    if verbose:
        print(f"\n[Step 2] PCA 降维: {states.shape[1]}D → {n_components}D")

    pca = PCA(n_components=n_components, random_state=0)
    states_pca = pca.fit_transform(states)

    if verbose:
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        print(f"[Step 2] 各主成分解释方差: {np.round(explained_var, 4)}")
        print(f"[Step 2] 累积解释方差: {cumulative_var[-1]:.4f} ({cumulative_var[-1]*100:.1f}%)")
        print(f"[Step 2] states_pca.shape = {states_pca.shape}")

    return states_pca, pca


def save_pca_results(states_pca, pca, output_dir: str):
    """保存 PCA 结果"""
    np.save(os.path.join(output_dir, "states_pca.npy"), states_pca)
    with open(os.path.join(output_dir, "pca_state.pkl"), "wb") as f:
        pickle.dump(pca, f)
    print(f"[Step 2] 已保存 states_pca.npy, pca_state.pkl → {output_dir}")


if __name__ == "__main__":
    import os
    # 独立测试 Step 2: 需要先从 Step 1 拿到 states
    OUTPUT_DIR = os.path.dirname(__file__)
    states_path = os.path.join(OUTPUT_DIR, "states.npy")
    if os.path.exists(states_path):
        states = np.load(states_path)
        states_pca, pca = reduce_states_pca(states)
        save_pca_results(states_pca, pca, OUTPUT_DIR)
    else:
        print(f"未找到 states.npy，请先运行 step1_extract.py")
