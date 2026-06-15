"""
Step 1: 从 LIBERO_OBJECT HDF5 文件中提取 states 和 actions

    state = concat([joint_states(7), ee_pos(3), ee_ori(3), gripper_states(2)]) = 15D
    action = actions (7D)

用法:
    from pca.step1_extract import extract_states_and_actions
    states, actions = extract_states_and_actions(DATA_DIR)
"""

import os
import glob
import numpy as np
import h5py


def extract_states_and_actions(data_dir: str, verbose: bool = True):
    """
    遍历所有 LIBERO_OBJECT HDF5 文件，提取 states 和 actions

    Args:
        data_dir: LIBERO_OBJECT 数据集目录路径
        verbose: 是否打印进度信息

    Returns:
        all_states  (N, 15)  — joint_pos(7) + ee_pos(3) + ee_ori(3) + gripper(2)
        all_actions (N, 7)
    """
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))

    if not hdf5_files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到任何 .hdf5 文件，请检查路径")

    if verbose:
        print(f"[Step 1] 找到 {len(hdf5_files)} 个 HDF5 文件")
        print(f"[Step 1] 数据集路径: {data_dir}")

    all_states = []
    all_actions = []
    total_demos = 0
    total_steps = 0

    for filepath in hdf5_files:
        fname = os.path.basename(filepath)
        demo_count = 0

        with h5py.File(filepath, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))

            for demo_key in demo_keys:
                demo = f[f"data/{demo_key}"]

                # 构造 state: joint_pos(7) + eef_pos(3) + eef_ori(3) + gripper_qpos(2) = 15D
                joint_pos = demo["obs/joint_states"][:]       # (T, 7)
                eef_pos   = demo["obs/ee_pos"][:]              # (T, 3)
                eef_ori   = demo["obs/ee_ori"][:]              # (T, 3)
                gripper   = demo["obs/gripper_states"][:]      # (T, 2)

                state = np.concatenate([joint_pos, eef_pos, eef_ori, gripper], axis=1)  # (T, 15)
                action = demo["actions"][:]                                              # (T, 7)

                all_states.append(state)
                all_actions.append(action)

                demo_count += 1
                total_steps += state.shape[0]

        total_demos += demo_count
        if verbose:
            print(f"  ✓ {fname}: {demo_count} demos")

    all_states  = np.concatenate(all_states, axis=0)   # (N, 15)
    all_actions = np.concatenate(all_actions, axis=0)   # (N, 7)

    if verbose:
        print(f"\n[Step 1] 总计: {total_demos} demos, {total_steps} steps")
        print(f"[Step 1] states.shape  = {all_states.shape}")
        print(f"[Step 1] actions.shape = {all_actions.shape}")
        print(f"[Step 1] state 特征构成: joint_pos(7) + ee_pos(3) + ee_ori(3) + gripper(2) = {all_states.shape[1]}D")

    return all_states, all_actions


if __name__ == "__main__":
    # 独立测试 Step 1
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "libero", "datasets", "libero_object")
    DATA_DIR = os.path.abspath(DATA_DIR)
    states, actions = extract_states_and_actions(DATA_DIR)
