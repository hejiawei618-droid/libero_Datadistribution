"""
LIBERO_OBJECT 数据分析脚本 — 统一入口
Step 1: 提取 states 和 actions  →  pca/step1_extract.py
Step 2: PCA 降维 states           →  pca/step2_pca.py

用法:
    python pca.py

输出 (保存在 pca/ 目录下):
    pca/states.npy       — 形状 (N, 15), 所有轨迹的 state 特征
    pca/actions.npy      — 形状 (N, 7),  所有轨迹的 action
    pca/states_pca.npy   — 形状 (N, 10), PCA 降维后的 states
    pca/pca_state.pkl    — 训练好的 PCA 对象 (sklearn)
"""

import os
import pickle
import numpy as np

from pca.step1_extract import extract_states_and_actions
from pca.step2_pca import reduce_states_pca

# ============================================================
# 配置
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "libero", "datasets", "libero_object")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "pca")
PCA_N_COMPONENTS = 10

# 数据过滤 (与 main.py CONFIG 保持一致)
TASK_FILTER = None          # "milk" / "butter" / None=全部
ACTION_DIRECTION = None     # "left" / "right" / None=全部
TEMPORAL_RATIO = None       # (0.0, 0.3) / None=全部帧
VISION_ENCODER = None       # "vit_small" / "resnet101" / None
VISION_CAMERAS = ["agentview"]  # ["agentview", "eye_in_hand"]


def main():
    data_dir = os.path.abspath(DATA_DIR)
    output_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    parts = [f"task={TASK_FILTER or 'ALL'}", f"action_dir={ACTION_DIRECTION or 'ALL'}"]
    if TEMPORAL_RATIO:
        parts.append(f"phase={TEMPORAL_RATIO[0]*100:.0f}%-{TEMPORAL_RATIO[1]*100:.0f}%")
    if VISION_ENCODER:
        parts.append(f"vision={VISION_ENCODER}+{VISION_CAMERAS}")
    desc = ", ".join(parts)
    print("=" * 60)
    print(f"LIBERO_OBJECT: Step 1 + Step 2  ({desc})")
    print("=" * 60)

    # Step 1: 提取数据 (支持过滤 + 视觉编码)
    states, actions = extract_states_and_actions(
        data_dir,
        task_name=TASK_FILTER,
        action_direction=ACTION_DIRECTION,
        temporal_ratio=TEMPORAL_RATIO,
        vision_encoder=VISION_ENCODER,
        vision_cameras=VISION_CAMERAS,
    )

    # Step 2: PCA 降维
    states_pca, pca = reduce_states_pca(states, n_components=PCA_N_COMPONENTS)

    # 保存到 pca/ 目录
    np.save(os.path.join(output_dir, "states.npy"), states)
    np.save(os.path.join(output_dir, "actions.npy"), actions)
    np.save(os.path.join(output_dir, "states_pca.npy"), states_pca)
    with open(os.path.join(output_dir, "pca_state.pkl"), "wb") as f:
        pickle.dump(pca, f)

    print(f"\n{'='*60}")
    print(f"保存文件 → {output_dir}/")
    print(f"  states.npy       → {states.shape}")
    print(f"  actions.npy      → {actions.shape}")
    print(f"  states_pca.npy   → {states_pca.shape}")
    print(f"  pca_state.pkl    → PCA 对象")
    print(f"{'='*60}")
    print("✅ Step 1 + Step 2 完成!")


if __name__ == "__main__":
    main()
