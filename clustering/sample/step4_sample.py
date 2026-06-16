"""
Step 4: 从 PCA 降维后的 states 中随机采样 500 个中心状态

    输入:  states_pca  (N, 10)
    输出:  sample_ids  (500,) — 随机选出的 500 个状态索引

用法:
    from sample.step4_sample import random_sample_states
    sample_ids = random_sample_states(states_pca, n_samples=500)
"""

import os
import numpy as np


def random_sample_states(states_pca: np.ndarray, n_samples: int = 500, random_seed: int = 0, verbose: bool = True):
    """
    从 states_pca 中随机无放回采样 n_samples 个中心状态

    Args:
        states_pca: (N, D) PCA 降维后的 state 特征
        n_samples: 采样数量, 默认 500
        random_seed: 随机种子, 保证可复现
        verbose: 是否打印信息

    Returns:
        sample_ids: (n_samples,) 随机选中的状态索引
    """
    N = len(states_pca)

    if n_samples > N:
        raise ValueError(f"n_samples ({n_samples}) 不能超过 states_pca 的样本数 ({N})")

    if verbose:
        print(f"[Step 4] 随机采样 Observation")
        print(f"[Step 4]   总状态数: {N}")
        print(f"[Step 4]   采样数量: {n_samples}")
        print(f"[Step 4]   随机种子: {random_seed}")

    rng = np.random.RandomState(random_seed)
    sample_ids = rng.choice(N, n_samples, replace=False)

    if verbose:
        print(f"[Step 4]   采样索引范围: [{sample_ids.min()}, {sample_ids.max()}]")
        print(f"[Step 4] 随机采样完成 ✓")

    return sample_ids


def save_sample_ids(sample_ids, output_dir: str, filename: str = "sample_ids.npy"):
    """保存采样索引到磁盘"""
    path = os.path.join(output_dir, filename)
    np.save(path, sample_ids)
    print(f"[Step 4] 保存 → {path}")


def load_sample_ids(output_dir: str, filename: str = "sample_ids.npy"):
    """加载采样索引"""
    path = os.path.join(output_dir, filename)
    return np.load(path)


if __name__ == "__main__":
    # 独立测试 Step 4
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    OUTPUT_DIR = os.path.dirname(__file__)

    states_pca_path = os.path.join(os.path.dirname(__file__), "..", "pca", "states_pca.npy")
    states_pca = np.load(states_pca_path)

    sample_ids = random_sample_states(states_pca, n_samples=100)
    save_sample_ids(sample_ids, OUTPUT_DIR)
