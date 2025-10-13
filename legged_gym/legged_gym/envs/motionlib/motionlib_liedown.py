import os
import math
import torch.utils
import yaml
import torch
import random
import pickle
from tqdm import tqdm

import copy
import torch
import numpy as np
from legged_gym.utils import torch_utils
from legged_gym.utils.math import wrap_to_pi

class MotionLib:
    def __init__(self, motion_file, mapping_file, dof_names, fps, device, window_length, ratio_random_range, thresh_robot2chair, near_mask):
        self.fps = fps
        self.device = device
        self.env_fps = 50
        self.window_length = window_length
        self.ratio_random_range = ratio_random_range
        self.thresh_robot2chair = thresh_robot2chair
        self.near_mask = near_mask
        self.end_effector_name = ["left_palm_link", "right_palm_link", "left_ankle_pitch_link", "right_ankle_pitch_link", "mid360_link"]
        self.load_motions(motion_file, mapping_file)
        self.process_motions(dof_names)

    def load_motions(self, motion_file, mapping_file):
        lines = open(mapping_file).readlines()
        lines = [line[:-1].split(" ") for line in lines]
        self.mapping = {k: int(v) for v, k in lines}

        ext = os.path.splitext(motion_file)[1]
        dir_name = os.path.dirname(motion_file)
        if (ext != ".yaml"):
            raise NotImplementedError
        
        with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
            motion_config = yaml.load(f, Loader=yaml.SafeLoader)

        self.motion_len = {}
        self.num_motion = {}
        self.tot_len = {}
        self.motion_start_ids = {}
        self.motion_end_ids = {}

        self.motion_data = {}
        self.motion_weights = {}
        self.motion_rsi_skipped_ranges = {}

        self.skills = list(motion_config['motions'].keys())

        tot_frames = 0
        for skill in self.skills:
            motion_list = motion_config['motions'][skill]
            self.motion_len[skill] = []
            self.motion_data[skill] = []
            self.motion_weights[skill] = []
            self.motion_rsi_skipped_ranges[skill] = []
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_data = torch.load(os.path.join(dir_name, curr_file))
                
                curr_motion_len = list(curr_data.values())[0].shape[0]
                curr_weight = motion_entry['weight']

                curr_rsi_skipped_range = motion_entry.get('rsi_skipped_range', [])
                if len(curr_rsi_skipped_range) == 0:
                    curr_rsi_skipped_range = [np.inf, -np.inf]
                else:
                    assert len(curr_rsi_skipped_range) == 2
                    assert curr_rsi_skipped_range[0] < curr_rsi_skipped_range[1]

                self.motion_len[skill].append(curr_motion_len)
                self.motion_data[skill].append(curr_data)
                self.motion_weights[skill].append(curr_weight)
                self.motion_rsi_skipped_ranges[skill].append(curr_rsi_skipped_range)

            self.motion_weights[skill] = torch.tensor(self.motion_weights[skill], dtype=torch.float, device=self.device)
            self.motion_rsi_skipped_ranges[skill] = torch.tensor(self.motion_rsi_skipped_ranges[skill], dtype=torch.float, device=self.device)
            self.motion_len[skill] = torch.tensor(self.motion_len[skill], dtype=torch.long, device=self.device)

            self.num_motion[skill] = self.motion_len[skill].shape[0]
            self.tot_len[skill] = self.motion_len[skill].sum()
            self.motion_end_ids[skill] = torch.cumsum(self.motion_len[skill], dim=0) + tot_frames
            self.motion_start_ids[skill] = torch.nn.functional.pad(self.motion_end_ids[skill], (1, -1), "constant", tot_frames)
            tot_frames += self.tot_len[skill]

        self.motion_weights_tot = torch.concat(list(self.motion_weights.values()), dim=0)
        self.motion_start_ids_tot = torch.concat(list(self.motion_start_ids.values()), dim=0)
        self.motion_end_ids_tot = torch.concat(list(self.motion_end_ids.values()), dim=0)
        self.motion_len_tot = torch.concat(list(self.motion_len.values()), dim=0)

    def process_motions(self, dof_names):
        self.tot_frames = sum(self.tot_len.values())

        self.motion_base_rpy = torch.zeros(self.tot_frames, 3, dtype=torch.float, device=self.device)
        self.motion_base_lin_vel = torch.zeros(self.tot_frames, 3, dtype=torch.float, device=self.device)
        self.motion_base_ang_vel = torch.zeros(self.tot_frames, 3, dtype=torch.float, device=self.device)
        self.motion_global_lin_vel = torch.zeros(self.tot_frames, 3, dtype=torch.float, device=self.device)
        self.motion_global_ang_vel = torch.zeros(self.tot_frames, 3, dtype=torch.float, device=self.device)
        self.motion_base_rot = torch.zeros(self.tot_frames, 6, dtype=torch.float, device=self.device)
        self.motion_dof_pos = torch.zeros(self.tot_frames, len(self.mapping), dtype=torch.float, device=self.device)
        self.motion_dof_vel = torch.zeros(self.tot_frames, len(self.mapping), dtype=torch.float, device=self.device)
        self.motion_base_height = torch.zeros(self.tot_frames, 1, dtype=torch.float, device=self.device)
        self.motion_base_quat = torch.zeros(self.tot_frames, 4, dtype=torch.float, device=self.device)
        self.motion_base_pos = torch.zeros(self.tot_frames, 3, dtype=torch.float, device=self.device)
        self.motion_chair_pos = torch.zeros(self.tot_frames, 3, dtype=torch.float, device=self.device)
        self.motion_chair_pos_global = torch.zeros(self.tot_frames, 3, dtype=torch.float, device=self.device)
        self.motion_chair_quat = torch.zeros(self.tot_frames, 4, dtype=torch.float, device=self.device)
        self.motion_end_effector_pos = torch.zeros(self.tot_frames, len(self.end_effector_name), 3, dtype=torch.float, device=self.device)
        self.motion_base_z_bias = torch.zeros(self.tot_frames, 1, dtype=torch.float, device=self.device)
        self.default_quat = torch.tensor([0., 0., 0., 1.], dtype=torch.float, device=self.device).view(1, 4)

        for skill in self.skills:
            for i, traj in enumerate(self.motion_data[skill]):
                start, end = self.motion_start_ids[skill][i], self.motion_end_ids[skill][i]
                self.motion_base_height[start:end] = traj["base_height"].reshape(-1, 1).clone().detach()
                self.motion_base_quat[start:end] = traj["base_quat"].clone().detach()
                self.motion_base_pos[start:end] = traj["base_position"].clone().detach()
                if skill == "lieDown":
                    traj_bias = -0.11
                    self.motion_base_pos[start:end, 2:3] += traj_bias

                self.motion_base_z_bias[start:end] = self.motion_base_pos[start:end, 2:3] - self.motion_base_height[start:end]

                self.motion_global_lin_vel[start:end-1] = (self.motion_base_pos[start+1:end] - self.motion_base_pos[start:end-1]) * self.fps
                self.motion_global_lin_vel[end-1:end] = self.motion_global_lin_vel[end-2:end-1]

                root_rot = traj["base_quat"].clone().detach()
                heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
                root_rot_obs = torch_utils.quat_mul(heading_rot, root_rot)
                self.motion_base_rot[start:end] = torch_utils.quat_to_tan_norm(root_rot_obs)

                self.motion_base_rpy[start:end] = euler_from_quaternion(traj["base_quat"]).clone().detach()
                rpy_unwrapped = self.motion_base_rpy.cpu().numpy()
                rpy_unwrapped = np.unwrap(rpy_unwrapped, axis=0)
                self.motion_global_ang_vel[start:end-1] = torch.tensor((rpy_unwrapped[start+1:end] - rpy_unwrapped[start:end-1]) * self.fps, dtype=torch.float, device=self.device)
                self.motion_global_ang_vel[end-1:end] = self.motion_global_ang_vel[end-2:end-1]

                dof_pos = traj["joint_position"].clone().detach()
                dof_vel = traj["joint_velocity"].clone().detach()
                for j, name in enumerate(dof_names):
                    if name in self.mapping.keys():
                        self.motion_dof_pos[start:end, j] = dof_pos[:, self.mapping[name]]
                        self.motion_dof_vel[start:end, j] = dof_vel[:, self.mapping[name]]

                for k, name in enumerate(self.end_effector_name):
                    self.motion_end_effector_pos[start:end, k] = traj["link_position"][:, k].clone().detach()
                
                if "chair_pos" in traj.keys():
                    self.motion_chair_pos_global[start:end] = traj["chair_pos"].clone().detach()
                    self.motion_chair_pos[start:end] = self.motion_chair_pos_global[start:end] - self.motion_base_pos[start:end]
                    self.motion_chair_quat[start:end] = traj["chair_quat"].clone().detach()
                elif skill == 'lieDown':
                    self.motion_chair_pos_global[start:end] = self.motion_base_pos[end-1:end].repeat(end-start, 1) # chose base_pos of last frame as the chair_pos
                    self.motion_chair_pos_global[start:end, 0] -= 0.2
                    self.motion_chair_pos_global[start:end, 2] += traj["chair_height_offset"]
                    self.motion_chair_pos[start:end] = self.motion_chair_pos_global[start:end] - self.motion_base_pos[start:end]
                    self.motion_chair_quat[start:end] = torch.tensor([0., 0., -1., 0.], dtype=torch.float, device=self.device).view(1, 4).repeat(end-start, 1)
                else:
                    self.motion_chair_pos[start:end] = traj["box_pos_local"].clone().detach()
                    self.motion_chair_pos_global[start:end] = self.motion_chair_pos[start:end] + self.motion_base_pos[start:end]
                    self.motion_chair_quat[start:end] = self.default_quat

        self.motion_base_lin_vel = torch_utils.quat_rotate_inverse(self.motion_base_quat, self.motion_global_lin_vel)
        self.motion_base_ang_vel = torch_utils.quat_rotate_inverse(self.motion_base_quat, self.motion_global_ang_vel)
        self.motion_end_effector_pos[:, 0, :] = torch_utils.quat_rotate_inverse(self.motion_base_quat, self.motion_end_effector_pos[:, 0, :])
        self.motion_end_effector_pos[:, 1, :] = torch_utils.quat_rotate_inverse(self.motion_base_quat, self.motion_end_effector_pos[:, 1, :])
        self.motion_end_effector_pos[:, 2, :] = torch_utils.quat_rotate_inverse(self.motion_base_quat, self.motion_end_effector_pos[:, 2, :])
        self.motion_end_effector_pos[:, 3, :] = torch_utils.quat_rotate_inverse(self.motion_base_quat, self.motion_end_effector_pos[:, 3, :])
        self.motion_end_effector_pos[:, 4, :] = torch_utils.quat_rotate_inverse(self.motion_base_quat, self.motion_end_effector_pos[:, 4, :])
        self.motion_chair_pos[:, :] = torch_utils.quat_rotate_inverse(self.motion_base_quat, self.motion_chair_pos[:, :])
        
        mask = torch.norm(self.motion_chair_pos[:, :2], dim=-1) > self.thresh_robot2chair
        directions = self.motion_chair_pos[mask, :2]
        norms = torch.norm(directions, dim=-1, keepdim=True)
        directions = directions / norms
        scaled_dirs = directions * self.thresh_robot2chair
        self.motion_chair_pos[mask] = torch.cat([scaled_dirs, torch.zeros((scaled_dirs.shape[0], 1), device=self.device, dtype=torch.float)], dim=1)
        # when the chair is too close to the robot, set the chair position to zero
        near_mask = torch.norm(self.motion_chair_pos[:, :2], dim=-1) <= self.near_mask
        self.motion_chair_pos[near_mask, :2] = 0.0

        self.motion_end_effector_pos = self.motion_end_effector_pos.view(self.tot_frames, -1)

        _, _, robot_yaw = torch_utils.euler_from_quaternion(self.motion_base_quat)
        _, _, chair_yaw = torch_utils.euler_from_quaternion(self.motion_chair_quat)
        self.motion_yaw_diff = wrap_to_pi(chair_yaw - robot_yaw).unsqueeze(-1)
        self.motion_yaw_diff[mask] = 0.

    def sample_motions(self, skill, num_envs):
        motion_ids = torch.multinomial(self.motion_weights[skill], num_envs, replacement=True)
        return motion_ids

    def sample_time_rsi(self, skill, motion_ids):
        n = motion_ids.shape[0]
        succ_sample_times = torch.zeros(n, dtype=torch.float, device=self.device)
        succ_sample_record = torch.zeros(n, dtype=torch.bool, device=self.device)
        while torch.sum(succ_sample_record) < n:
            mask = (succ_sample_record == False)
            phase = torch.rand(motion_ids[mask].shape, device=self.device)
            motion_len = self.motion_len[skill][motion_ids[mask]]
            motion_time = phase * (motion_len - 1)
            skipped_range = self.motion_rsi_skipped_ranges[skill][motion_ids[mask]]
            curr_num_samples = motion_time.shape[0]
            curr_succ = torch.where(
                torch.logical_and(motion_time >= skipped_range[:, 0], motion_time <= skipped_range[:, 1]),
                torch.zeros(curr_num_samples, dtype=torch.bool, device=self.device),
                torch.ones(curr_num_samples, dtype=torch.bool, device=self.device),
            )
            curr_succ_inds = torch.nonzero(mask)[curr_succ, 0]
            succ_sample_times[curr_succ_inds] = motion_time[curr_succ]
            succ_sample_record[curr_succ_inds] = True

        return succ_sample_times
    
    def get_motion_state(self, skill, motion_ids, motion_times):
        start_ids = self.motion_start_ids[skill][motion_ids]
        floors, ceils = torch.floor(motion_times).long(), torch.ceil(motion_times).long()
        time0, time1 = start_ids+floors, start_ids+ceils
        w0, w1 = (1-(motion_times-floors)).reshape(-1, 1), (motion_times-floors).reshape(-1, 1)

        def blend_motion(motion):
            motion0, motion1 = motion[time0], motion[time1]
            return w0 * motion0 + w1 * motion1
        
        root_pos = blend_motion(self.motion_base_pos)
        if skill == "loco":
            root_pos[:, 2:3] = root_pos[:, 2:3] - blend_motion(self.motion_base_z_bias) + 0.05
        root_lin_vel = blend_motion(self.motion_global_lin_vel)
        root_ang_vel = blend_motion(self.motion_global_ang_vel)
        dof_pos = blend_motion(self.motion_dof_pos)
        dof_vel = blend_motion(self.motion_dof_vel)
        ee_pos = blend_motion(self.motion_end_effector_pos)

        root_quat_0 = self.motion_base_quat[start_ids+floors]
        root_quat_1 = self.motion_base_quat[start_ids+ceils]
        dot = torch.sum(root_quat_0 * root_quat_1, dim=-1, keepdim=True)
        root_quat_0 = torch.where(dot < 0, -root_quat_0, root_quat_0)
        root_quat = w0 * root_quat_0 + w1 * root_quat_1
        root_quat = torch.nn.functional.normalize(root_quat, dim=-1)
        
        return root_pos, root_quat, root_lin_vel, root_ang_vel, dof_pos, dof_vel, ee_pos
    
    def get_obj_motion_state(self, skill, motion_ids, motion_times):
        start_ids = self.motion_start_ids[skill][motion_ids]
        floors, ceils = torch.floor(motion_times).long(), torch.ceil(motion_times).long()
        time0, time1 = start_ids+floors, start_ids+ceils
        w0, w1 = (1-(motion_times-floors)).reshape(-1, 1), (motion_times-floors).reshape(-1, 1)

        def blend_motion(motion):
            motion0, motion1 = motion[time0], motion[time1]
            return w0 * motion0 + w1 * motion1
        
        chair_pos = blend_motion(self.motion_chair_pos_global)
        if skill == "loco":
            chair_pos[:, 2:3] = chair_pos[:, 2:3] - blend_motion(self.motion_base_z_bias) + 0.05
        chair_quat_0 = self.motion_chair_quat[start_ids+floors]
        chair_quat_1 = self.motion_chair_quat[start_ids+ceils]
        dot = torch.sum(chair_quat_0 * chair_quat_1, dim=-1, keepdim=True)
        chair_quat_0 = torch.where(dot < 0, -chair_quat_0, chair_quat_0)
        chair_quat = w0 * chair_quat_0 + w1 * chair_quat_1
        chair_quat = torch.nn.functional.normalize(chair_quat, dim=-1)
        chair_quat = torch_utils.calc_heading_quat(chair_quat)
        
        return chair_pos, chair_quat
    
    def get_amp_hist_obs(self, skill, motion_ids, motion_times):
        start_ids = self.motion_start_ids[skill][motion_ids]

        ratio = self.fps / self.env_fps
        amp_obs = torch.empty((motion_times.shape[0], 0), device=self.device)

        offsets = torch.arange(self.window_length-1, -1, -1, device=self.device).float() * ratio
        times = motion_times[:, None] - offsets[None, :]
        floor = torch.floor(times).long()
        ceil = floor + 1
        floor = torch.clamp(floor, min=0)
        ceil = torch.clamp(ceil, min=0, max=self.tot_frames-1)
        lerp_ratio = (times % 1.0).unsqueeze(-1)

        start_ids_exp = start_ids[:, None]
        floor_idx = start_ids_exp + floor
        ceil_idx = start_ids_exp + ceil

        def gather_and_interp(tensor):
            f = tensor[floor_idx]
            c = tensor[ceil_idx]
            return f * (1 - lerp_ratio) + c * lerp_ratio
        
        def gather_and_interp_angle(tensor):
            a = tensor[floor_idx]
            b = tensor[ceil_idx]
            diff = wrap_to_pi(b - a)
            return wrap_to_pi(a + lerp_ratio * diff)

        motion_dof_pos = gather_and_interp(self.motion_dof_pos)
        base_height = gather_and_interp(self.motion_base_height)
        motion_dof_pos = gather_and_interp(self.motion_dof_pos)
        end_effector_pos = gather_and_interp(self.motion_end_effector_pos)
        base_lin_vel = gather_and_interp(self.motion_base_lin_vel)
        base_ang_vel = gather_and_interp(self.motion_base_ang_vel)
        base_rot = gather_and_interp(self.motion_base_rot)
        chair_pos = gather_and_interp(self.motion_chair_pos)
        chair_yaw = gather_and_interp_angle(self.motion_yaw_diff)
        
        amp_obs = torch.cat([base_height, motion_dof_pos, end_effector_pos, base_lin_vel, base_ang_vel, base_rot, chair_pos, chair_yaw], dim=-1).reshape(motion_times.shape[0], -1)

        return amp_obs
    
    @staticmethod    
    def calc_blend(motion, time0, time1, w0, w1):
        motion0, motion1 = motion[time0], motion[time1]
        new_w0 = w0.reshape(w0.shape + (1,) * (motion0.dim() - w0.dim()))
        new_w1 = w1.reshape(w1.shape + (1,) * (motion1.dim() - w1.dim()))
        return new_w0 * motion0 + new_w1 * motion1
    
    def get_expert_obs(self, batch_size):
        motion_ids = torch.multinomial(self.motion_weights_tot, batch_size, replacement=True)
        start_ids = self.motion_start_ids_tot[motion_ids]
        end_ids = self.motion_end_ids_tot[motion_ids]
        motion_len = self.motion_len_tot[motion_ids]

        time_in_proportion = torch.rand(batch_size).to(self.device)
        clip_tail_proportion = (self.window_length * self.fps / self.env_fps + 2) / motion_len
        time_in_proportion = time_in_proportion.clamp(torch.zeros_like(clip_tail_proportion).to(self.device), 1 - clip_tail_proportion)

        motion_ids = start_ids + time_in_proportion * (end_ids - start_ids)
        amp_obs = torch.zeros((batch_size, 0), device=self.device)

        def gather_and_interp_angle(tensor, floor_idx, ceil_idx, lerp_ratio):
            a = tensor[floor_idx]
            b = tensor[ceil_idx]
            diff = wrap_to_pi(b - a)
            return wrap_to_pi(a + lerp_ratio * diff)

        ratio = self.fps / self.env_fps
        ratio *= np.random.uniform(self.ratio_random_range[0], self.ratio_random_range[1])
        for i in range(self.window_length):
            floor = torch.floor(motion_ids + i * ratio).long()
            ceil = floor + 1
            linear_ratio = ((motion_ids + i * ratio) % 1).reshape(-1, 1)
            base_height_next = self.motion_base_height[floor] * (1 - linear_ratio) + self.motion_base_height[ceil] * linear_ratio
            motion_dof_pos_next = self.motion_dof_pos[floor] * (1 - linear_ratio) + self.motion_dof_pos[ceil] * linear_ratio
            end_effector_pos_next = self.motion_end_effector_pos[floor] * (1 - linear_ratio) + self.motion_end_effector_pos[ceil] * linear_ratio
            base_lin_vel_next = self.motion_base_lin_vel[floor] * (1 - linear_ratio) + self.motion_base_lin_vel[ceil] * linear_ratio
            base_ang_vel_next = self.motion_base_ang_vel[floor] * (1 - linear_ratio) + self.motion_base_ang_vel[ceil] * linear_ratio
            base_rot_next = self.motion_base_rot[floor] * (1 - linear_ratio) + self.motion_base_rot[ceil] * linear_ratio
            chair_pos_next = self.motion_chair_pos[floor] * (1 - linear_ratio) + self.motion_chair_pos[ceil] * linear_ratio
            chair_yaw_next = gather_and_interp_angle(self.motion_yaw_diff, floor, ceil, linear_ratio)

            amp_obs = torch.cat([amp_obs, base_height_next, motion_dof_pos_next, end_effector_pos_next, base_lin_vel_next, base_ang_vel_next, base_rot_next, chair_pos_next, chair_yaw_next], dim=-1).view(batch_size, -1)
            
        return amp_obs


def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return torch.cat([roll_x.view(-1, 1), pitch_y.view(-1, 1), yaw_z.view(-1, 1)], dim=1)