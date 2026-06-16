"""
LIBERO_OBJECT 完整聚类分析流程 — 一键运行 Step 1 ~ Step 8

用法:
    cd clustering/
    python run_all.py

也可单独运行每个步骤:
    python pca.py          # Step 1+2 : 提取数据 + PCA state
    python knn.py           # Step 3   : kNN 索引
    python sample.py        # Step 4   : 随机采样
    python pca_action.py    # Step 5+6 : 邻域 action + PCA action
    python gmm.py           # Step 7+8 : GMM 峰数 + 统计

修改参数: 直接编辑下方 CONFIG 区域即可
======================================================================
流程:
    LIBERO HDF5
        ↓ Step 1: 提取 states (15D) + actions (7D)
    states.npy, actions.npy
        ↓ Step 2: PCA → states_pca (10D)
    states_pca.npy
        ↓ Step 3: kNN 索引 (n_neighbors=N_NEIGHBORS)
    knn_index.pkl
        ↓ Step 4: 随机采样 N_SAMPLES 个中心状态
    sample_ids.npy
        ↓ Step 5: 对每个中心取 N_NEIGHBORS 邻域 action
        ↓ Step 6: PCA(action, 7D→3D)
    local_actions_pca.npy
        ↓ Step 7: GMM(K=1~MAX_K) + BIC 选峰
        ↓ Step 8: 统计 K* 分布
    all_best_k.npy, kstar_summary.json
    1111222
======================================================================
"""

import os
import sys
import time
import pickle
import numpy as np

# ============================================================
# ⚙️  CONFIG — 在此修改所有参数
# ============================================================
CONFIG = {
    # Step 1: 数据过滤
    #   task_name: 只提取指定任务 (如 "milk", "butter", "tomato_sauce"), None = 全部
    #   action_direction: action 方向 ("left"=dy<0 向左, "right"=dy>0 向右), None = 全部
    #   temporal_ratio: 每个 demo 只取指定时间比例 (如 (0.0, 0.3)=前30%接近物体阶段)
    #   vision_encoder: 视觉编码器 ("vit_small"/"resnet101"), None = 只用 robot state
    #   vision_cameras: 使用哪些相机视角 (["agentview", "eye_in_hand"])
    "TASK_FILTER": "butter",                          # 只选牛奶任务
    "ACTION_DIRECTION": None,                     # 只选向左的 action
    "TEMPORAL_RATIO": (0, 0.3),                  # 只取每个 demo 前 30% 帧
    "VISION_ENCODER": "vit_small",                  # ViT-small → 384D / None=不用视觉
    "VISION_CAMERAS": ["agentview", "eye_in_hand"], # 使用两个相机

    # Step 3: kNN 邻域大小
    "N_NEIGHBORS": 50,

    # Step 4: 随机采样中心状态数
    "N_SAMPLES": 200,

    # Step 2: state PCA 目标维度
    "PCA_STATE_N_COMPONENTS": 100,

    # Step 6: action PCA 目标维度
    "PCA_ACTION_N_COMPONENTS": 7,

    # Step 7: GMM K 搜索范围 [k_min, k_max+1)
    "GMM_K_RANGE": (1, 6),  # K = 1, 2, 3, 4, 5

    # 随机种子 (保证可复现)
    "RANDOM_SEED": 0,
}
# ============================================================

# 确保 clustering 目录在 sys.path 中
CLUSTERING_DIR = os.path.dirname(os.path.abspath(__file__))
if CLUSTERING_DIR not in sys.path:
    sys.path.insert(0, CLUSTERING_DIR)


def run_pca():
    """Step 1 + Step 2: 提取数据 + PCA state"""
    from pca.step1_extract import extract_states_and_actions
    from pca.step2_pca import reduce_states_pca

    DATA_DIR = os.path.join(CLUSTERING_DIR, "..", "libero", "datasets", "libero_object")
    OUTPUT_DIR = os.path.join(CLUSTERING_DIR, "pca")

    task_filter = CONFIG.get("TASK_FILTER")
    action_dir = CONFIG.get("ACTION_DIRECTION")
    temporal_r = CONFIG.get("TEMPORAL_RATIO")
    v_encoder  = CONFIG.get("VISION_ENCODER")
    v_cameras  = CONFIG.get("VISION_CAMERAS")
    desc_parts = [f"task={task_filter or 'ALL'}", f"dir={action_dir or 'ALL'}"]
    if temporal_r:
        desc_parts.append(f"phase={temporal_r[0]*100:.0f}%-{temporal_r[1]*100:.0f}%")
    if v_encoder:
        desc_parts.append(f"vision={v_encoder}+{v_cameras}")
    desc = ", ".join(desc_parts)
    print("\n" + "=" * 60)
    print(f"  Step 1: 提取 states & actions  ({desc})")
    print("=" * 60)
    states, actions = extract_states_and_actions(
        os.path.abspath(DATA_DIR),
        task_name=task_filter,
        action_direction=action_dir,
        temporal_ratio=temporal_r,
        vision_encoder=v_encoder,
        vision_cameras=v_cameras,
    )

    print("\n" + "=" * 60)
    print(f"  Step 2: PCA 降维 states (15D → {CONFIG['PCA_STATE_N_COMPONENTS']}D)")
    print("=" * 60)
    states_pca, pca = reduce_states_pca(
        states, n_components=CONFIG["PCA_STATE_N_COMPONENTS"]
    )

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "states.npy"), states)
    np.save(os.path.join(OUTPUT_DIR, "actions.npy"), actions)
    np.save(os.path.join(OUTPUT_DIR, "states_pca.npy"), states_pca)
    with open(os.path.join(OUTPUT_DIR, "pca_state.pkl"), "wb") as f:
        pickle.dump(pca, f)

    print(f"  ✓ states={states.shape}, actions={actions.shape}, states_pca={states_pca.shape}")
    return states_pca, actions


def run_knn():
    """Step 3: 建立 kNN 索引"""
    from knn.step3_knn import build_knn_index, save_knn

    OUTPUT_DIR = os.path.join(CLUSTERING_DIR, "knn")
    states_pca = np.load(os.path.join(CLUSTERING_DIR, "pca", "states_pca.npy"))
    k = CONFIG["N_NEIGHBORS"]

    print("\n" + "=" * 60)
    print(f"  Step 3: 建立 kNN 索引 (k={k})")
    print("=" * 60)
    knn = build_knn_index(states_pca, n_neighbors=k)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_knn(knn, OUTPUT_DIR)

    print(f"  ✓ kNN(k={k}) 索引已建立")
    return knn


def run_sample():
    """Step 4: 随机采样中心状态"""
    from sample.step4_sample import random_sample_states, save_sample_ids

    OUTPUT_DIR = os.path.join(CLUSTERING_DIR, "sample")
    states_pca = np.load(os.path.join(CLUSTERING_DIR, "pca", "states_pca.npy"))
    n = CONFIG["N_SAMPLES"]

    print("\n" + "=" * 60)
    print(f"  Step 4: 随机采样 {n} 个中心状态")
    print("=" * 60)
    sample_ids = random_sample_states(
        states_pca, n_samples=n, random_seed=CONFIG["RANDOM_SEED"]
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_sample_ids(sample_ids, OUTPUT_DIR)

    print(f"  ✓ 采样 {len(sample_ids)} 个状态")
    return sample_ids


def run_pca_action():
    """Step 5 + Step 6: 邻域 action + PCA action"""
    from pca_action.step5_local_actions import get_local_actions_for_samples, save_local_actions
    from pca_action.step6_pca_action import reduce_actions_pca, save_actions_pca

    OUTPUT_DIR = os.path.join(CLUSTERING_DIR, "pca_action")
    n_neighbors = CONFIG["N_NEIGHBORS"]
    n_components = CONFIG["PCA_ACTION_N_COMPONENTS"]

    # 加载依赖
    sample_ids = np.load(os.path.join(CLUSTERING_DIR, "sample", "sample_ids.npy"))
    states_pca = np.load(os.path.join(CLUSTERING_DIR, "pca", "states_pca.npy"))
    actions    = np.load(os.path.join(CLUSTERING_DIR, "pca", "actions.npy"))
    with open(os.path.join(CLUSTERING_DIR, "knn", "knn_index.pkl"), "rb") as f:
        knn = pickle.load(f)

    print("\n" + "=" * 60)
    print(f"  Step 5: 获取邻域 Action (k={n_neighbors})")
    print("=" * 60)
    local_actions_list, neighbor_ids = get_local_actions_for_samples(
        sample_ids, knn, states_pca, actions, n_neighbors=n_neighbors
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_local_actions(local_actions_list, neighbor_ids, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"  Step 6: PCA 压缩 Action (7D → {n_components}D)")
    print("=" * 60)
    local_actions_pca_list, act_pca_models = reduce_actions_pca(
        local_actions_list, n_components=n_components
    )
    save_actions_pca(local_actions_pca_list, act_pca_models, OUTPUT_DIR)

    n_samples = len(sample_ids)
    print(f"  ✓ local_actions_pca: {n_samples} × ({n_neighbors}, {n_components})")
    return local_actions_pca_list


def run_gmm():
    """Step 7 + Step 8: GMM 峰数判断 + 统计"""
    from gmm.step7_gmm import gmm_select_peaks, save_gmm_results
    from gmm.step8_statistics import final_statistics, save_statistics

    OUTPUT_DIR = os.path.join(CLUSTERING_DIR, "gmm")
    k_range = CONFIG["GMM_K_RANGE"]

    # 加载 local_actions_pca
    local_actions_pca = np.load(
        os.path.join(CLUSTERING_DIR, "pca_action", "local_actions_pca.npy")
    )
    local_actions_pca_list = list(local_actions_pca)

    print("\n" + "=" * 60)
    print(f"  Step 7: GMM 峰数判断 (BIC, K={k_range[0]}~{k_range[1]-1})")
    print("=" * 60)
    all_best_k, bic_curves = gmm_select_peaks(
        local_actions_pca_list,
        k_range=k_range,
        random_seed=CONFIG["RANDOM_SEED"],
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_gmm_results(all_best_k, bic_curves, None, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  Step 8: 统计汇总")
    print("=" * 60)
    counter, summary = final_statistics(all_best_k)
    save_statistics(counter, summary, OUTPUT_DIR)

    return counter, summary


# ============================================================
# 主入口
# ============================================================
def main():
    t_start = time.time()

    print("=" * 60)
    print("  LIBERO_OBJECT 完整聚类分析 — Step 1 ~ Step 8")
    print("=" * 60)
    print(f"  工作目录: {CLUSTERING_DIR}")
    print(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    tf = CONFIG.get("TASK_FILTER") or "ALL"
    ad = CONFIG.get("ACTION_DIRECTION") or "ALL"
    tr = CONFIG.get("TEMPORAL_RATIO")
    tr_str = f"{tr[0]*100:.0f}%-{tr[1]*100:.0f}%" if tr else "ALL"
    ve = CONFIG.get("VISION_ENCODER") or "none"
    print(f"  配置: TASK={tf}, ACTION_DIR={ad}, PHASE={tr_str}, VISION={ve}, "
          f"N_NEIGHBORS={CONFIG['N_NEIGHBORS']}, "
          f"N_SAMPLES={CONFIG['N_SAMPLES']}, "
          f"GMM_K_RANGE={CONFIG['GMM_K_RANGE']}")

    # Step 1+2
    t0 = time.time()
    run_pca()
    print(f"  ⏱  Step 1+2 耗时: {time.time() - t0:.1f}s")

    # Step 3
    t0 = time.time()
    run_knn()
    print(f"  ⏱  Step 3 耗时: {time.time() - t0:.1f}s")

    # Step 4
    t0 = time.time()
    run_sample()
    print(f"  ⏱  Step 4 耗时: {time.time() - t0:.1f}s")

    # Step 5+6
    t0 = time.time()
    run_pca_action()
    print(f"  ⏱  Step 5+6 耗时: {time.time() - t0:.1f}s")

    # Step 7+8
    t0 = time.time()
    run_gmm()
    print(f"  ⏱  Step 7+8 耗时: {time.time() - t0:.1f}s")

    # 总耗时
    print(f"\n{'='*60}")
    print(f"  总耗时: {time.time() - t_start:.1f}s")
    print(f"  结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print("  ✅ LIBERO_OBJECT 完整聚类分析完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
