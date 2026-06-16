"""
Step 7: GMM 判断峰数 (核心步骤)

对每个采样点的 K 个 PCA action (3D) 拟合 GMM (K=1~5),
用 BIC 准则选择最佳峰数 K*

    输入:  local_actions_pca_list  [K] 每个 (N, 3)
    输出:  all_best_k              (K,) 每个采样点的最佳 K*
           bic_curves              [K] 每个采样点的 BIC 曲线 (5,)

用法:
    from gmm.step7_gmm import gmm_select_peaks
    all_best_k, bic_curves = gmm_select_peaks(local_actions_pca_list)
"""

import os
import pickle
import numpy as np
from collections import Counter
from sklearn.mixture import GaussianMixture


def gmm_select_peaks(
    local_actions_pca_list: list,
    k_range: tuple = (1, 6),  # [1, 6) → K=1,2,3,4,5
    random_seed: int = 0,
    verbose: bool = True,
):
    """
    对每个采样点的 PCA action 做 GMM+BIC 选峰

    Args:
        local_actions_pca_list: list of (300, D) arrays
        k_range: (k_min, k_max+1), 默认 (1, 6) → K=1..5
        random_seed: 随机种子
        verbose: 是否打印信息

    Returns:
        all_best_k: (K_samples,) 每个采样点的最佳峰数
        bic_curves: list of (len_k_range,) arrays — BIC 曲线
    """
    K_samples = len(local_actions_pca_list)
    k_values = list(range(k_range[0], k_range[1]))  # [1, 2, 3, 4, 5]

    if verbose:
        print(f"[Step 7] GMM 峰数判断 (BIC 准则)")
        print(f"[Step 7]   采样点数: {K_samples}")
        print(f"[Step 7]   K 搜索范围: {k_values}")
        print(f"[Step 7]   PCA action 维度: {local_actions_pca_list[0].shape}")

    all_best_k = []
    bic_curves = []

    for i, local_actions_pca in enumerate(local_actions_pca_list):
        bic_scores = []

        for k in k_values:
            gmm = GaussianMixture(n_components=k, random_state=random_seed)
            gmm.fit(local_actions_pca)
            bic_scores.append(gmm.bic(local_actions_pca))

        bic_scores = np.array(bic_scores)
        best_k = int(np.argmin(bic_scores)) + k_range[0]

        all_best_k.append(best_k)
        bic_curves.append(bic_scores)

        if verbose and (i + 1) % 20 == 0:
            # 打印当前进度和分布
            counter = Counter(all_best_k)
            print(f"[Step 7]   {i + 1}/{K_samples} 完成 | 当前分布: {dict(sorted(counter.items()))}")

    all_best_k = np.array(all_best_k)

    if verbose:
        print(f"[Step 7] GMM 峰数判断完成 ✓")

    return all_best_k, bic_curves


def summarize_peaks(all_best_k: np.ndarray, verbose: bool = True):
    """
    统计最佳峰数分布

    Args:
        all_best_k: (K_samples,) 最佳峰数
        verbose: 是否打印

    Returns:
        counter: Counter 对象
        stats:   dict with counts and percentages
    """
    counter = Counter(all_best_k)
    stats = {}
    total = len(all_best_k)

    if verbose:
        print(f"\n{'='*40}")
        print(f"Step 7 统计结果")
        print(f"{'='*40}")
        print(f"{'K*':<8} {'Count':<10} {'Percent':<10}")
        print(f"{'-'*28}")

    for k in sorted(counter.keys()):
        pct = counter[k] / total * 100
        stats[k] = {"count": counter[k], "percent": round(pct, 1)}
        if verbose:
            print(f"{k:<8} {counter[k]:<10} {pct:.1f}%")

    if verbose:
        print(f"{'-'*28}")
        print(f"单峰 (K=1):  {counter.get(1, 0)} ({counter.get(1,0)/total*100:.1f}%)")
        print(f"双峰 (K=2):  {counter.get(2, 0)} ({counter.get(2,0)/total*100:.1f}%)")
        print(f"三峰+(K≥3):  {sum(counter.get(k,0) for k in range(3, 10))} "
              f"({sum(counter.get(k,0) for k in range(3,10))/total*100:.1f}%)")

    return counter, stats


def save_gmm_results(all_best_k, bic_curves, counter, output_dir: str, verbose: bool = True):
    """保存 GMM 结果"""
    np.save(os.path.join(output_dir, "all_best_k.npy"), all_best_k)

    # bic_curves 堆叠为 (K_samples, 5)
    bic_stacked = np.stack(bic_curves, axis=0)
    np.save(os.path.join(output_dir, "bic_curves.npy"), bic_stacked)

    # BIC 最小值对应对数
    k_values = np.arange(1, bic_stacked.shape[1] + 1)
    best_k_per_sample = k_values[np.argmin(bic_stacked, axis=1)]
    np.save(os.path.join(output_dir, "best_k_per_sample.npy"), best_k_per_sample)

    # 保存 counter (用于 Step 8 可视化)
    with open(os.path.join(output_dir, "kstar_counter.pkl"), "wb") as f:
        pickle.dump(counter, f)

    if verbose:
        print(f"[Step 7] 保存 → {output_dir}/")
        print(f"  all_best_k.npy        → {all_best_k.shape}")
        print(f"  bic_curves.npy         → {bic_stacked.shape}")
        print(f"  best_k_per_sample.npy  → {best_k_per_sample.shape}")
        print(f"  kstar_counter.pkl      → Counter")


def load_gmm_results(output_dir: str):
    """加载 GMM 结果"""
    all_best_k = np.load(os.path.join(output_dir, "all_best_k.npy"))
    bic_stacked = np.load(os.path.join(output_dir, "bic_curves.npy"))
    with open(os.path.join(output_dir, "kstar_counter.pkl"), "rb") as f:
        counter = pickle.load(f)
    return all_best_k, list(bic_stacked), counter


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    OUTPUT_DIR = os.path.dirname(__file__)

    # 加载 local_actions_pca
    local_actions_pca = np.load(
        os.path.join(os.path.dirname(__file__), "..", "pca_action", "local_actions_pca.npy")
    )
    local_actions_pca_list = list(local_actions_pca)

    all_best_k, bic_curves = gmm_select_peaks(local_actions_pca_list)
    counter, stats = summarize_peaks(all_best_k)
    save_gmm_results(all_best_k, bic_curves, counter, OUTPUT_DIR)
