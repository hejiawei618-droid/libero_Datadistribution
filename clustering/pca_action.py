"""
LIBERO_OBJECT 数据分析脚本 — Step 5+6 统一入口
Step 5: 获取邻域 Action       →  pca_action/step5_local_actions.py
Step 6: PCA 压缩 Action       →  pca_action/step6_pca_action.py

用法:
    python pca_action.py

输出 (保存在 pca_action/ 目录下):
    pca_action/local_actions.npy       — (500, 1000, 7)   500 个采样点各 1000 个邻域 action
    pca_action/neighbor_ids.npy        — (500, 1000)      对应的 kNN 邻域索引
    pca_action/local_actions_pca.npy   — (500, 1000, 3)   PCA 压缩后的 action
    pca_action/act_pca_models.pkl      — [500]            每个采样点的 PCA 模型
"""

import os
import pickle
import numpy as np

from pca_action.step5_local_actions import get_local_actions_for_samples, save_local_actions
from pca_action.step6_pca_action import reduce_actions_pca, save_actions_pca

# ============================================================
# 配置
# ============================================================
SAMPLE_IDS_PATH   = os.path.join(os.path.dirname(__file__), "sample", "sample_ids.npy")
STATES_PCA_PATH   = os.path.join(os.path.dirname(__file__), "pca", "states_pca.npy")
ACTIONS_PATH      = os.path.join(os.path.dirname(__file__), "pca", "actions.npy")
KNN_INDEX_PATH    = os.path.join(os.path.dirname(__file__), "knn", "knn_index.pkl")
OUTPUT_DIR        = os.path.join(os.path.dirname(__file__), "pca_action")
N_NEIGHBORS       = 1000  # 可在 run_all.py 中统一修改
PCA_N_COMPONENTS  = 3


def main():
    output_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("LIBERO_OBJECT: Step 5 (邻域 Action) + Step 6 (PCA 压缩 Action)")
    print("=" * 60)

    # 加载依赖数据
    sample_ids = np.load(os.path.abspath(SAMPLE_IDS_PATH))
    states_pca = np.load(os.path.abspath(STATES_PCA_PATH))
    actions    = np.load(os.path.abspath(ACTIONS_PATH))
    with open(os.path.abspath(KNN_INDEX_PATH), "rb") as f:
        knn = pickle.load(f)

    print(f"加载: sample_ids={sample_ids.shape}, states_pca={states_pca.shape}, actions={actions.shape}")

    # Step 5: 获取邻域 Action
    local_actions_list, neighbor_ids = get_local_actions_for_samples(
        sample_ids, knn, states_pca, actions, n_neighbors=N_NEIGHBORS
    )
    save_local_actions(local_actions_list, neighbor_ids, output_dir)

    # Step 6: PCA 压缩 Action
    local_actions_pca_list, act_pca_models = reduce_actions_pca(
        local_actions_list, n_components=PCA_N_COMPONENTS
    )
    save_actions_pca(local_actions_pca_list, act_pca_models, output_dir)

    print(f"\n{'='*60}")
    print(f"保存文件 → {output_dir}/")
    print(f"  local_actions.npy       → ({len(sample_ids)}, {N_NEIGHBORS}, 7)")
    print(f"  neighbor_ids.npy        → ({len(sample_ids)}, {N_NEIGHBORS})")
    print(f"  local_actions_pca.npy   → ({len(sample_ids)}, {N_NEIGHBORS}, 3)")
    print(f"  act_pca_models.pkl      → {len(sample_ids)} 个 PCA 模型")
    print(f"{'='*60}")
    print("✅ Step 5 + Step 6 完成!")


if __name__ == "__main__":
    main()
