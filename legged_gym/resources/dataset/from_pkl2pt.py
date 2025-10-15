import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def compute_velocity(data, fps):
    """Compute finite-difference velocities"""
    vel = np.zeros_like(data)
    vel[1:] = (data[1:] - data[:-1]) * fps
    vel[0] = vel[1]  # copy first frame
    return vel


def process_pkl_to_pt(pkl_path, save_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    fps = data["fps"]
    root_pos = np.asarray(data["root_pos"])
    root_rot = np.asarray(data["root_rot"])
    dof_pos = np.asarray(data["dof_pos"])
    local_body_pos = np.asarray(data["local_body_pos"])   # (T, N, 3)
    body_names = data.get("link_body_list", [])

    T = root_pos.shape[0]
    # ---------- Base ----------
    base_position = root_pos
    base_quat = root_rot  # (w, x, y, z) expected
    base_height = root_pos[:, 2]

    # ---------- Velocities ----------
    base_linear_velocity = compute_velocity(base_position, fps)
    joint_velocity = compute_velocity(dof_pos, fps)

    # angular vel from quaternion difference
    base_angular_velocity = np.zeros_like(base_position)
    for i in range(1, T):
        dq = R.from_quat(base_quat[i, [1, 2, 3, 0]]) * R.from_quat(base_quat[i-1, [1, 2, 3, 0]]).inv()
        angvel = dq.as_rotvec() * fps
        base_angular_velocity[i] = angvel
    base_angular_velocity[0] = base_angular_velocity[1]

    # ---------- Link positions ----------
    link_position = local_body_pos  # (T, N, 3)

    # ---------- Torch ----------
    motion_dict = {
        "base_height": torch.from_numpy(base_height).float(),
        "base_position": torch.from_numpy(base_position).float(),
        "base_quat": torch.from_numpy(base_quat).float(),
        "base_linear_velocity": torch.from_numpy(base_linear_velocity).float(),
        "base_angular_velocity": torch.from_numpy(base_angular_velocity).float(),
        "joint_position": torch.from_numpy(dof_pos).float(),
        "joint_velocity": torch.from_numpy(joint_velocity).float(),
        "link_position": torch.from_numpy(link_position).float(),
        "link_names": body_names,
        "fps": fps,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(motion_dict, save_path)
    print(f"[OK] Saved {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="folder of pkl files")
    parser.add_argument("--output_root", required=True, help="output folder for pt")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    pkl_files = []
    for root, _, files in os.walk(args.input_root):
        for f in files:
            if f.endswith(".pkl"):
                pkl_files.append(os.path.join(root, f))

    print(f"Found {len(pkl_files)} pkl files.")
    for pkl_path in tqdm(pkl_files):
        rel = os.path.relpath(pkl_path, args.input_root)
        pt_path = os.path.join(args.output_root, rel).replace(".pkl", ".pt")
        process_pkl_to_pt(pkl_path, pt_path)


if __name__ == "__main__":
    main()
