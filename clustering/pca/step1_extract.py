"""
Step 1: 从 LIBERO_OBJECT HDF5 文件中提取 states 和 actions

    state = concat([robot_state(15D), vision_features(D_v)])

    视觉编码:
        agentview_rgb (128×128) → ViT → 384D 语义特征
        eye_in_hand_rgb (128×128) → ViT → 384D 语义特征

    替代 flatten RGB + PCA，采用机器人论文标准做法。

用法:
    from pca.step1_extract import extract_states_and_actions
    states, actions = extract_states_and_actions(
        DATA_DIR,
        task_name="milk",
        action_direction="left",
        temporal_ratio=(0.0, 0.3),
        vision_encoder="vit_small",    # None = 只用 robot state
        vision_cameras=["agentview", "eye_in_hand"],
    )
"""

import os
import glob
import numpy as np
import h5py


def extract_states_and_actions(
    data_dir: str,
    task_name: str = None,
    action_direction: str = None,
    temporal_ratio: tuple = None,
    vision_encoder: str = None,
    vision_cameras: list = None,
    verbose: bool = True,
):
    """
    遍历 LIBERO_OBJECT HDF5 文件，提取 states 和 actions

    Args:
        data_dir:         LIBERO_OBJECT 数据集目录路径
        task_name:        任务名过滤 (如 "milk"), None = 全部
        action_direction: action 方向过滤 ("left"/"right"), None = 全部
        temporal_ratio:   时间窗口比例 (start, end), None = 全部帧
        vision_encoder:   视觉编码器名称, None = 只用 robot state
                          "vit_small" → ViT-small 384D
                          "resnet101" → ResNet101 2048D
        vision_cameras:   使用哪些相机, 默认 ["agentview"]
                         可选: ["agentview", "eye_in_hand"]
        verbose:          是否打印进度信息

    Returns:
        all_states  (N, D)  — robot_state(15) + vision_features
        all_actions (N, 7)
    """
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))

    if not hdf5_files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到任何 .hdf5 文件，请检查路径")

    # 任务名过滤
    if task_name:
        hdf5_files = [f for f in hdf5_files if task_name in os.path.basename(f)]
        if not hdf5_files:
            available = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))]
            raise FileNotFoundError(
                f"未找到包含 '{task_name}' 的任务文件。\n可用任务: "
                f"{[n.replace('_demo.hdf5','').replace('pick_up_the_','').replace('_and_place_it_in_the_basket','') for n in available]}"
            )

    # --- 参数验证 ---
    if action_direction and action_direction not in ("left", "right"):
        raise ValueError(f"action_direction 必须是 'left' 或 'right', 收到: '{action_direction}'")

    if temporal_ratio:
        if not (isinstance(temporal_ratio, (tuple, list)) and len(temporal_ratio) == 2):
            raise ValueError(f"temporal_ratio 必须是 (start, end) 元组")
        if not (0.0 <= temporal_ratio[0] < temporal_ratio[1] <= 1.0):
            raise ValueError(f"temporal_ratio 必须满足 0 <= start < end <= 1")

    if vision_cameras is None:
        vision_cameras = []

    # --- 初始化视觉编码器 (先打印配置再加载模型) ---
    if verbose:
        print(f"[Step 1] 找到 {len(hdf5_files)} 个 HDF5 文件" + (
            f" (过滤: task='{task_name}')" if task_name else ""
        ))
        print(f"[Step 1] 数据集路径: {data_dir}")
        if action_direction:
            print(f"[Step 1] Action 方向过滤: {'dy < 0 (← 左)' if action_direction == 'left' else 'dy > 0 (→ 右)'}")
        if temporal_ratio:
            print(f"[Step 1] 时间窗口: 每个 demo 的 {temporal_ratio[0]*100:.0f}%~{temporal_ratio[1]*100:.0f}% 帧")

    encoder = None
    vision_dim = 0
    if vision_encoder and vision_cameras:
        from .vision_encoder import VisionEncoder
        encoder = VisionEncoder(_get_model_name(vision_encoder))
        vision_dim = encoder.feature_dim
        if verbose:
            print(f"[Step 1] 视觉编码: {vision_encoder} → {vision_dim}D × {len(vision_cameras)} cameras")
    elif verbose:
        print(f"[Step 1] 视觉编码: 无 (仅 robot state)")

    # --- 构建 state 描述 ---
    state_parts = ["joint_pos(7)", "ee_pos(3)", "ee_ori(3)", "gripper(2)"]
    for cam in vision_cameras:
        state_parts.append(f"{cam}({vision_dim}D)")
    robot_dim = 15
    total_dim = robot_dim + vision_dim * len(vision_cameras)

    all_states = []
    all_actions = []
    total_demos = 0
    total_steps = 0
    total_filtered = 0

    for filepath in hdf5_files:
        fname = os.path.basename(filepath)
        demo_count = 0

        with h5py.File(filepath, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))

            for demo_key in demo_keys:
                demo = f[f"data/{demo_key}"]

                # --- robot state (15D) ---
                joint_pos = demo["obs/joint_states"][:]
                eef_pos   = demo["obs/ee_pos"][:]
                eef_ori   = demo["obs/ee_ori"][:]
                gripper   = demo["obs/gripper_states"][:]
                robot_state = np.concatenate([joint_pos, eef_pos, eef_ori, gripper], axis=1)
                action = demo["actions"][:]                    # (T, 7)
                T_full = len(robot_state)

                # --- 时间窗口过滤 (先过滤 robot + action, 减少后续图像处理量) ---
                if temporal_ratio:
                    t_start = int(T_full * temporal_ratio[0])
                    t_end   = int(T_full * temporal_ratio[1])
                    robot_state = robot_state[t_start:t_end]
                    action      = action[t_start:t_end]

                # --- Action 方向过滤 ---
                keep_mask = np.ones(len(robot_state), dtype=bool)
                if action_direction:
                    dy = action[:, 1]
                    keep_mask = (dy < 0) if action_direction == "left" else (dy > 0)
                    total_filtered += (~keep_mask).sum()

                robot_state = robot_state[keep_mask]
                action      = action[keep_mask]

                if len(robot_state) == 0:
                    continue

                # --- 视觉编码 (只处理保留的帧, 节省 GPU 时间) ---
                if encoder and vision_cameras:
                    # 读取原始图像: 从完整 demo 中按 temporal_ratio + action_direction 取
                    # 注意: action_direction mask 不能直接应用于原始 T_full 长度的图像
                    # 所以重新读取完整图像，然后应用相同的过滤
                    t_start_img = int(T_full * temporal_ratio[0]) if temporal_ratio else 0
                    t_end_img   = int(T_full * temporal_ratio[1]) if temporal_ratio else T_full

                    vision_features = []
                    for cam in vision_cameras:
                        cam_key = f"{cam}_rgb"
                        img = demo[f"obs/{cam_key}"][t_start_img:t_end_img]  # uint8
                        img = img[keep_mask]  # 同步过滤
                        feat = encoder.encode(img)  # (T_filtered, D_v)
                        vision_features.append(feat)

                    state = np.concatenate([robot_state] + vision_features, axis=1)
                else:
                    state = robot_state

                all_states.append(state)
                all_actions.append(action)
                demo_count += 1
                total_steps += state.shape[0]

        total_demos += demo_count
        if verbose:
            info = f"  ✓ {fname}: {demo_count} demos"
            if action_direction:
                info += " (过滤后)"
            print(info)

    all_states  = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    if verbose:
        print(f"\n[Step 1] 总计: {total_demos} demos, {total_steps} steps")
        print(f"[Step 1] states.shape  = {all_states.shape}")
        print(f"[Step 1] actions.shape = {all_actions.shape}")
        if action_direction and total_filtered > 0:
            print(f"[Step 1] 方向过滤: 剔除了 {total_filtered} 帧")
        print(f"[Step 1] state = {' + '.join(state_parts)} = {all_states.shape[1]}D")

    return all_states, all_actions


def _get_model_name(name: str) -> str:
    """视觉编码器名称 → timm 模型名"""
    mapping = {
        "vit_small":  "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "vit_base":   "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "resnet101":  "resnet101.tv_in1k",
    }
    if name in mapping:
        return mapping[name]
    return name  # 直接传 timm 模型名也可


if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "libero", "datasets", "libero_object")
    DATA_DIR = os.path.abspath(DATA_DIR)
    states, actions = extract_states_and_actions(DATA_DIR)
