# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
#
# 修改要点（KISS）：
# 1) 不再用 gymutil 画 marker；改为把 marker 当作 URDF actor。
# 2) 统一用多 actor 根状态张量：self.all_root_states.view(num_envs, num_actors_per_env, 13)
#    槽位：0=机器人，1..S=轨迹采样点，S+1=目标点（S=self._num_traj_samples）
# 3) 每步在 _update_traj_markers() 中一次性写回所有 marker 的 root state。
# 4) markers 创建时使用“机器人初始化位置”的 x,y（第一帧就对齐）。
# 5) 关闭 marker 碰撞（shape.filter = 0xFFFF），并让 marker 高度保持常数（不随 base 高度变化）。
#
# 需要在 LeggedRobotCfg 里新增（示例）：
# class MarkerCfg:
#     class asset:
#         file = "{LEGGED_GYM_ROOT_DIR}/resources/markers/small_sphere.urdf"
#         name = "traj_marker"
#         self_collisions = 0
#     disable_gravity = True
#     height_offset = 0.05   # 相对 base 初始 z 的抬升高度
# marker = MarkerCfg()

import os
import time
import copy
import numpy as np

import torch
import torch.nn.functional as F

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.torch_utils import (
    calc_heading_quat_inv,
    quat_to_tan_norm,
    euler_from_quaternion,
    quat_from_angle_axis,
    torch_rand_float,
)
from legged_gym.utils import traj_generator

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.motionlib.motionlib_styleloco import MotionLib


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)

        # 先读取本体观测相关（不含任务观测）
        self.num_one_step_proprio_obs = self.cfg.env.num_one_step_proprio_obs
        self.actor_history_length = self.cfg.env.num_actor_history
        self.actor_obs_length = self.cfg.env.num_actor_obs
        self.num_privileged_obs = self.cfg.env.num_privileged_obs

        # —— 轨迹任务参数与开关（与 HumanoidTraj 对齐）——
        self.enable_traj_task = hasattr(self.cfg, "traj")
        if self.enable_traj_task:
            self._num_traj_samples = int(self.cfg.traj.num_samples)
            self._traj_sample_timestep = float(self.cfg.traj.sample_timestep)
            self._traj_speed_min = float(self.cfg.traj.speed_min)
            self._traj_speed_max = float(self.cfg.traj.speed_max)
            self._traj_accel_max = float(self.cfg.traj.accel_max)
            self._sharp_turn_prob = float(self.cfg.traj.sharp_turn_prob)
            self._sharp_turn_angle = float(self.cfg.traj.sharp_turn_angle)
            self._traj_fail_dist = float(self.cfg.traj.fail_dist)
            self._traj_num_verts = int(self.cfg.traj.num_vertices)
            self._traj_dtheta_max = float(self.cfg.traj.dtheta_max)
        self.multi_task_cfg = getattr(self.cfg, "multi_task", None)
        assert self.multi_task_cfg is not None, "multi_task config required"
        self.task_names = list(self.multi_task_cfg.task_names)
        self.task_name_to_id = {name: idx for idx, name in enumerate(self.task_names)}
        self.num_tasks = len(self.task_names)
        self.task_init_prob = torch.tensor(self.multi_task_cfg.task_init_prob, dtype=torch.float)
        self.task_obs_dim = dict(self.multi_task_cfg.task_obs_dim)

        self.num_task_obs = int(self.cfg.env.num_task_obs)
        self._enable_task_mask_obs = bool(getattr(self.cfg.env, "enable_task_mask_obs", False))
        total_task_obs_dim = sum(int(self.task_obs_dim.get(name, 0)) for name in self.task_names)
        if self._enable_task_mask_obs:
            total_task_obs_dim += self.num_tasks
        assert (
            self.num_task_obs == total_task_obs_dim
        ), "num_task_obs must match declared task observation dimensions and mask settings"

        # 单步 actor 观测 = 本体 + 任务
        self.num_one_step_actor_obs = self.num_one_step_proprio_obs + self.num_task_obs

        self._sit_cfg = getattr(self.multi_task_cfg, "sitdown")
        self._carry_cfg = getattr(self.multi_task_cfg, "carrybox")
        self._stand_cfg = getattr(self.multi_task_cfg, "standup")

        # —— 新增：marker 高度偏移（默认 0.0，可在 cfg.marker.height_offset 配置）——
        self._marker_height_offset = float(getattr(getattr(self.cfg, "marker", object()), "height_offset", 0.0))

        # 每个 env 的常数 marker z（在 _create_envs 中写入）
        self._marker_z0 = None  # torch.tensor(num_envs, device=self.device) after _create_envs

        # 会调用 create_sim -> _create_envs
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        # Build metadata for multi-task observation organization
        self._build_multi_task_metadata()

        self.task_init_prob = self.task_init_prob / torch.clamp(self.task_init_prob.sum(), min=1e-6)
        self.task_init_prob = self.task_init_prob.to(self.device)
        self.task_indicator = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.task_mask = torch.zeros(self.num_envs, self.num_tasks, dtype=torch.bool, device=self.device)

        self._sit_target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._sit_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._sit_target_facing = torch.zeros(self.num_envs, 2, device=self.device)
        self._sit_target_height = torch.zeros(self.num_envs, 1, device=self.device)

        self._carry_box_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._carry_box_goal = torch.zeros(self.num_envs, 3, device=self.device)
        self._carry_box_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self._carry_box_start = torch.zeros(self.num_envs, 3, device=self.device)

        self._stand_target_height = torch.full((self.num_envs, 1), float(self._stand_cfg.target_height), device=self.device)
        self._stand_up_dir = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)
        self._stand_marker_pos = torch.zeros(self.num_envs, 3, device=self.device)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        if self.enable_traj_task:
            self._build_traj_generator()
        else:
            self._traj_gen = None

        self._prepare_reward_function()
        self.num_amp_obs = cfg.amp.num_obs

        self.init_done = True
        self.amp_obs_buf = torch.zeros(self.num_envs, self.num_amp_obs, device=self.device, dtype=torch.float)

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_multi_task(env_ids)

    # ---------------- Core loop ----------------
    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs, amp_obs_buf = self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs, amp_obs_buf

    def play_dataset_step(self, time_idx):
        time_idx = time_idx % self.motionlib.motion_base_pos.shape[0]
        for env_id, _ in enumerate(self.envs):
            self.root_states[env_id, 0:3] = self.motionlib.motion_base_pos[time_idx]
            self.root_states[env_id, 3:7] = self.motionlib.motion_base_quat[time_idx]
            self.root_states[env_id, 7:10] = self.motionlib.motion_global_lin_vel[time_idx]
            self.root_states[env_id, 10:13] = self.motionlib.motion_global_ang_vel[time_idx]
            self.dof_pos[env_id] = self.motionlib.motion_dof_pos[time_idx]
            self.dof_vel[env_id] = self.motionlib.motion_dof_vel[time_idx]

        # 简化：直接整体写回
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.all_root_states.reshape(-1, 13)))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self._refresh_sim_tensors()
        self.render()
        self.common_step_counter += 1
        self.gym.simulate(self.sim)

    def render(self, sync_frame_time=True):
        super().render(sync_frame_time)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _build_multi_task_metadata(self):
        """Pre-compute indexing and mask info for multi-task observations."""

        each_sizes = [int(self.task_obs_dim.get(name, 0)) for name in self.task_names]
        cumulative = torch.tensor([0] + each_sizes, device=self.device, dtype=torch.long).cumsum(dim=0)
        total_dim = int(cumulative[-1].item()) if len(cumulative) > 0 else 0

        if total_dim > 0:
            mask = torch.zeros(self.num_tasks, total_dim, device=self.device, dtype=torch.bool)
            for idx in range(self.num_tasks):
                start, end = cumulative[idx].item(), cumulative[idx + 1].item()
                if end > start:
                    mask[idx, start:end] = True
        else:
            mask = torch.zeros(self.num_tasks, 0, device=self.device, dtype=torch.bool)

        self._each_subtask_obs_size = each_sizes
        self._task_obs_indices = cumulative
        self._task_obs_total_dim = total_dim
        self._task_obs_mask = mask
        self._multi_task_info = {
            "onehot_size": self.num_tasks,
            "tota_subtask_obs_size": total_dim,
            "each_subtask_obs_size": each_sizes,
            "each_subtask_obs_mask": mask,
            "each_subtask_obs_indx": cumulative,
            "enable_task_mask_obs": self._enable_task_mask_obs,
            "each_subtask_name": self.task_names,
        }

    # --------------- Observations / Rewards ---------------
    def compute_amp_observations(self):
        base_height_l = self.root_states[:, 2] - self.feet_pos[:, 0, 2]
        base_height_r = self.root_states[:, 2] - self.feet_pos[:, 1, 2]
        base_height = torch.max(base_height_l, base_height_r).unsqueeze(-1)

        dof_pos = self.dof_pos[:, self.amp_obs_joint_id].clone()
        dof_vel = self.dof_vel[:, self.amp_obs_joint_id].clone()
        base_lin_vel = self.base_lin_vel.clone()
        base_ang_vel = self.base_ang_vel.clone()

        heading_rot = calc_heading_quat_inv(self.base_quat)
        root_rot_obs = quat_mul(heading_rot, self.base_quat)
        root_rot_obs = quat_to_tan_norm(root_rot_obs)

        current_amp_obs = torch.cat((base_height, dof_pos, self.end_effector_pos, base_lin_vel, base_ang_vel, root_rot_obs), dim=-1)
        self.amp_obs_buf = torch.cat((self.amp_obs_buf[:, self.cfg.amp.num_one_step_obs:], current_amp_obs), dim=-1)
        return self.amp_obs_buf.clone()

    def post_physics_step(self):
        self._refresh_sim_tensors()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        self.base_lin_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,7:10])
        self.base_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,10:13])

        self.end_effector_pos = torch.concatenate((self.rigid_body_states[:, self.hand_pos_indices[0], :3],
                                                  self.rigid_body_states[:, self.hand_pos_indices[1], :3],
                                                  self.feet_pos[:, 0], self.feet_pos[:, 1],
                                                  self.rigid_body_states[:, self.head_index, :3]), dim=-1)
        self.end_effector_pos = self.end_effector_pos - self.root_states[:, :3].repeat(1, 5)
        for i in range(5):
            self.end_effector_pos[:, 3*i: 3*i+3] = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index, 3:7], self.end_effector_pos[:, 3*i: 3*i+3])

        self.projected_gravity[:] = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index, 3:7], self.gravity_vec)

        self.feet_pos[:] = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel[:] = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.pelvis_contact_pos = self.rigid_body_states[:, self.pelvis_contact_index, :3]

        if self.enable_traj_task:
            self._update_traj_target()

        self.left_feet_pos = self.rigid_body_states[:, self.left_feet_indices, 0:3]
        self.right_feet_pos = self.rigid_body_states[:, self.right_feet_indices, 0:3]

        # contacts
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) * self.contact_filt
        self.feet_air_time += self.dt

        # 关节功率历史
        joint_powers = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((joint_powers, self.joint_powers[:, :-1]), dim=1)

        self._post_physics_step_callback()

        self.check_termination()
        self.compute_reward()

        amp_obs_buf = self.compute_amp_observations()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)

        self.reset_idx(env_ids)
        self.compute_observations()

        self.disturbance[:, :, :] = 0.0
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        self.feet_air_time *= ~self.contact_filt

        # 用 URDF marker：每步把采样点和目标点一次性写回
        if self.enable_traj_task:
            self._update_traj_markers()

        return env_ids, termination_privileged_obs, amp_obs_buf

    def check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length

        self.reset_buf |= torch.logical_or(torch.abs(self.roll)>0.5, torch.abs(self.pitch-0.25)>0.85)
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.root_states[:, 2] < 0.35
        if self.enable_traj_task:
            traj_dist = torch.norm(self._traj_curr_target[:, :2] - self.root_states[:, 0:2], dim=-1)
            self.reset_buf |= traj_dist > self._traj_fail_dist

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        if self.cfg.env.action_curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_action_curriculum(env_ids)

        self._reset_actors(env_ids)
        self._resample_commands(env_ids)
        # 简化：整体写回
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.all_root_states.reshape(-1, 13)))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        if self.enable_traj_task:
            self._reset_traj_follow_task(env_ids)

        self._reset_multi_task(env_ids)

        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.joint_powers[env_ids] = 0.
        self.delay_buffer[:, env_ids, :] = self.dof_pos[env_ids] - self.default_dof_pos
        self.reset_buf[env_ids] = 1

        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.delay:
            self.delay_idx[env_ids] = torch.randint(low=0, high=self.cfg.domain_rand.max_delay_timesteps, size=(len(env_ids), ), device=self.device)

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.action_curriculum:
            self.extras["episode"]["action_curriculum_ratio"] = self.action_curriculum_ratio
        self.episode_length_buf[env_ids] = 0

        if self.enable_traj_task:
            self._update_traj_markers()

    def compute_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if torch.isnan(rew).any():
                print(name)
                import ipdb; ipdb.set_trace()
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        current_obs = torch.cat((self.commands[:, :3],
                                 self.base_ang_vel  * self.obs_scales.ang_vel,
                                 self.projected_gravity,
                                 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                 self.dof_vel * self.obs_scales.dof_vel,
                                 self.end_effector_pos,
                                 self.actions,
                                 self.base_lin_vel * self.obs_scales.lin_vel,
                                 ), dim=-1)

        current_actor_obs = torch.clone(current_obs[:,:-3])

        if self.add_noise:
            # 只对 actor 的本体/动作部分加噪声；任务观测不加噪声
            current_actor_obs = current_actor_obs + (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[0:(9 + 2 * self.num_dof + self.num_actions + 15)]

        if self.enable_traj_task:
            task_obs_actor, task_obs_critic = self.compute_task_observations()
            self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_actor_obs:], current_actor_obs, task_obs_actor), dim=-1)
            self.privileged_obs_buf = torch.cat((current_obs, task_obs_critic), dim=-1)
        else:
            self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_proprio_obs:], current_actor_obs), dim=-1)
            self.privileged_obs_buf = current_obs.clone()

        self.extras["task_mask"] = self.task_mask
        self.extras["task_uid"] = self.task_indicator

    def compute_termination_observations(self, env_ids):
        current_obs = torch.cat((self.commands[:, :3],
                                 self.base_ang_vel  * self.obs_scales.ang_vel,
                                 self.projected_gravity,
                                 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                 self.dof_vel * self.obs_scales.dof_vel,
                                 self.end_effector_pos,
                                 self.actions,
                                 self.base_lin_vel * self.obs_scales.lin_vel,
                                 ), dim=-1)

        if self.enable_traj_task:
            _, task_obs_critic = self.compute_task_observations()
            return torch.cat((current_obs, task_obs_critic), dim=-1)[env_ids]

        return current_obs[env_ids]

    def compute_task_observations(self):
        traj_obs = torch.zeros(self.num_envs, self.task_obs_dim.get("traj", 0), device=self.device, dtype=self.root_states.dtype)
        if self.enable_traj_task and self.task_obs_dim.get("traj", 0) > 0:
            traj_samples = self._fetch_traj_samples()
            self._traj_samples_buf[:] = traj_samples
            traj_obs = compute_location_observations(self.root_states, traj_samples)

        sit_obs = self._compute_sit_obs()
        carry_obs = self._compute_carry_obs()
        stand_obs = self._compute_stand_obs()

        chunk_map = {
            "traj": traj_obs,
            "sitdown": sit_obs,
            "carrybox": carry_obs,
            "standup": stand_obs,
        }

        task_chunks = []
        for name in self.task_names:
            dim = int(self.task_obs_dim.get(name, 0))
            if dim <= 0:
                continue
            chunk = chunk_map.get(name)
            if chunk is None or chunk.shape[1] != dim:
                chunk = torch.zeros(self.num_envs, dim, device=self.device, dtype=self.root_states.dtype)
            task_chunks.append(chunk)

        if len(task_chunks) > 0:
            task_obs_real = torch.cat(task_chunks, dim=-1)
        else:
            task_obs_real = torch.zeros(self.num_envs, 0, device=self.device, dtype=self.root_states.dtype)

        if self._enable_task_mask_obs and task_obs_real.numel() > 0 and self._task_obs_mask.numel() > 0:
            mask = torch.matmul(
                self.task_mask.to(dtype=task_obs_real.dtype),
                self._task_obs_mask.to(dtype=task_obs_real.dtype),
            )
            task_obs_real = task_obs_real * mask

        if self._enable_task_mask_obs:
            task_obs = torch.cat((task_obs_real, self.task_mask.float()), dim=-1)
        else:
            task_obs = task_obs_real

        self.traj_obs_buf = task_obs_real
        return task_obs, task_obs

    def get_multi_task_info(self):
        """Expose structured multi-task observation metadata for policy construction."""

        return self._multi_task_info

    # ---------------- Sim creation ----------------
    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        print("*"*80)
        print("Start creating ground...")
        start = time.time()
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time.time() - start))
        print("*"*80)
        self._create_envs()

    def set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # -------------- Callbacks / props --------------
    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], self.cfg.domain_rand.restitution_range[1], (self.num_envs,1), device=self.device)
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
        return props

    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            for i in range(len(rigid_shape_props)):
                if self.cfg.domain_rand.randomize_friction:
                    rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_restitution:
                    rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_payload_mass:
            props[self.torso_link_index].mass = self.default_rigid_body_mass[self.torso_link_index] + self.payload[env_id, 0]
        if self.cfg.domain_rand.randomize_com_displacement:
            props[self.torso_link_index].com = self.default_com_torso + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(0, len(props)):
                if i == self.torso_link_index:
                    pass
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]
        return props

    def _post_physics_step_callback(self):
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            if self.cfg.commands.heading_to_ang_vel:
                forward = quat_apply(self.base_quat, self.forward_vec)
                heading = torch.atan2(forward[:, 1], forward[:, 0])
                self.ori_error = wrap_to_pi(self.commands[:, 3] - heading)
                self.commands[:, 2] = torch.clip(0.5 * self.ori_error, self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1])
            else:
                self.commands[:, 2] = self.commands[:, 3]

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def _resample_commands(self, env_ids):
        if env_ids.numel() == 0:
            return
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] *= torch.abs(self.commands[env_ids, 2]) > 0.2

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        if self.cfg.domain_rand.delay:
            self.delay_buffer = torch.concatenate((self.delay_buffer[1:], actions_scaled.unsqueeze(0)), dim=0)
            self.joint_pos_target = self.default_dof_pos + self.delay_buffer[self.delay_idx, torch.arange(len(self.delay_idx)), :]
        else:
            self.joint_pos_target = self.default_dof_pos + actions_scaled

        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        self.computed_torques = torques * self.motor_strength + self.actuation_offset
        return torch.clip(self.computed_torques, -self.torque_limits, self.torque_limits)

    # -------------- Resets --------------
    def _reset_actors(self, env_ids):
        if self.cfg.asset.reset_mode == 'default':
            self._reset_default(env_ids)
        elif self.cfg.asset.reset_mode == 'random':
            self._reset_ref_state_init(env_ids)
        elif self.cfg.asset.reset_mode == 'hybrid':
            self._reset_hybrid_state_init(env_ids)
        else:
            raise NotImplementedError

    def _reset_default(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)

        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dof_pos = self.default_dof_pos * torch_rand_float(self.cfg.domain_rand.initial_joint_pos_scale[0], self.cfg.domain_rand.initial_joint_pos_scale[1], (len(env_ids), self.num_dof), device=self.device)
            init_dof_pos += torch_rand_float(self.cfg.domain_rand.initial_joint_pos_offset[0], self.cfg.domain_rand.initial_joint_pos_offset[1], (len(env_ids), self.num_dof), device=self.device)
            self.dof_pos[env_ids] = torch.clip(init_dof_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch.ones((len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

    def _reset_ref_state_init(self, env_ids):
        sk_ids = torch.multinomial(self.skill_init_prob, num_samples=env_ids.shape[0], replacement=True)
        for uid, sk_name in enumerate(self.skill):
            curr_env_ids = env_ids[(sk_ids == uid).nonzero().squeeze(-1)]
            if len(curr_env_ids) == 0:
                continue
            num_envs = len(curr_env_ids)
            motion_ids = self.motionlib.sample_motions(sk_name, num_envs)
            motion_times = self.motionlib.sample_time_rsi(sk_name, motion_ids)
            root_pos, root_rot, root_lin_vel, root_ang_vel, dof_pos, dof_vel, ee_pos = self.motionlib.get_motion_state(sk_name, motion_ids, motion_times)

            self.root_states[curr_env_ids, :3] = root_pos + self.env_origins[curr_env_ids]
            self.root_states[curr_env_ids, 3:7] = root_rot
            self.root_states[curr_env_ids, 7:10] = root_lin_vel
            self.root_states[curr_env_ids, 10:13] = root_ang_vel
            self.root_states[curr_env_ids, 7:13] = torch_rand_float(-0.2, 0.2, (len(curr_env_ids), 6), device=self.device)

            dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
            dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
            if self.cfg.domain_rand.randomize_initial_joint_pos:
                init_dof_pos = dof_pos * torch_rand_float(self.cfg.domain_rand.initial_joint_pos_scale[0], self.cfg.domain_rand.initial_joint_pos_scale[1], (len(curr_env_ids), self.num_dof), device=self.device)
                init_dof_pos += torch_rand_float(self.cfg.domain_rand.initial_joint_pos_offset[0], self.cfg.domain_rand.initial_joint_pos_offset[1], (len(curr_env_ids), self.num_dof), device=self.device)
                self.dof_pos[curr_env_ids] = torch.clip(init_dof_pos, dof_lower, dof_upper)
            else:
                self.dof_pos[curr_env_ids] = dof_pos * torch.ones((len(curr_env_ids), self.num_dof), device=self.device)
            self.dof_vel[curr_env_ids] = dof_vel

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self.cfg.asset.hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0
        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)
        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

    # ---------------- Traj gen / markers ----------------
    def _build_traj_generator(self):
        episode_dur = self.max_episode_length * self.dt
        self._traj_gen = traj_generator.TrajGenerator(
            self.num_envs, episode_dur, self._traj_num_trajs_vertices(), self.device,
            self._traj_dtheta_max, self._traj_speed_min, self._traj_speed_max,
            self._traj_accel_max, self._sharp_turn_prob, self._sharp_turn_angle
        )
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        init_pos = self.env_origins + self.base_init_state[:3]
        self._traj_gen.reset(env_ids, init_pos)
        self._traj_samples_buf = torch.zeros(self.num_envs, self._num_traj_samples, 3, dtype=torch.float, device=self.device, requires_grad=False)
        time_zero = torch.zeros_like(env_ids, dtype=torch.float)
        self._traj_curr_target = self._traj_gen.calc_pos(env_ids, time_zero)
        self._traj_samples_buf[:] = self._fetch_traj_samples()

    def _traj_num_trajs_vertices(self):
        return self._traj_num_verts

    def _update_traj_target(self):
        if not self.enable_traj_task:
            return
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        time_now = self.episode_length_buf.to(torch.float) * self.dt
        self._traj_curr_target = self._traj_gen.calc_pos(env_ids, time_now)

    def _fetch_traj_samples(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        timestep_beg = self.episode_length_buf[env_ids].to(torch.float) * self.dt
        timesteps = torch.arange(self._num_traj_samples, device=self.device, dtype=torch.float) * self._traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps
        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)
        traj_samples_flat = self._traj_gen.calc_pos(env_ids_tiled.flatten(), traj_timesteps.flatten())
        traj_samples = traj_samples_flat.view(env_ids.shape[0], self._num_traj_samples, -1)
        return traj_samples

    def _reset_traj_follow_task(self, env_ids):
        if env_ids.numel() == 0:
            return

        # 机器人回到初始位姿
        root_pos = self.env_origins[env_ids] + self.base_init_state[:3]
        ars = self.all_root_states.view(self.num_envs, self.num_actors_per_env, 13)
        ars[env_ids, 0, 0:3] = root_pos
        ars[env_ids, 0, 3:7] = self.base_init_state[3:7]
        ars[env_ids, 0, 7:13] = 0.0

        # 轨迹 reset
        self._traj_gen.reset(env_ids, root_pos)
        t0 = torch.zeros_like(env_ids, dtype=torch.float)
        self._traj_curr_target[env_ids] = self._traj_gen.calc_pos(env_ids, t0)
        self._traj_samples_buf[env_ids] = self._fetch_traj_samples(env_ids)

        # z 使用固定常数：每个 env 独立的 marker_z0
        marker_z = self._marker_z0[env_ids]
        quat_identity = torch.tensor([0, 0, 0, 1], dtype=torch.float, device=self.device).view(1, 4)

        # 采样点槽位：1..S
        for k in range(self._num_traj_samples):
            pos = self._traj_samples_buf[env_ids, k, :].clone()
            ars[env_ids, 1 + k, 0] = pos[:, 0]            # x
            ars[env_ids, 1 + k, 1] = pos[:, 1]            # y
            ars[env_ids, 1 + k, 2] = marker_z             # z 固定
            ars[env_ids, 1 + k, 3:7] = quat_identity.repeat(len(env_ids), 1)
            ars[env_ids, 1 + k, 7:13] = 0.0

        # 目标点槽位：S+1
        slot = 1 + self._num_traj_samples
        tgt = self._traj_curr_target[env_ids].clone()
        ars[env_ids, slot, 0] = tgt[:, 0]
        ars[env_ids, slot, 1] = tgt[:, 1]
        ars[env_ids, slot, 2] = marker_z
        ars[env_ids, slot, 3:7] = quat_identity.repeat(len(env_ids), 1)
        ars[env_ids, slot, 7:13] = 0.0

        # 简化：整体写回
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.all_root_states.reshape(-1, 13)))

    def _reset_multi_task(self, env_ids):
        if env_ids.numel() == 0:
            return

        sampled = torch.multinomial(self.task_init_prob, env_ids.numel(), replacement=True)
        self.task_indicator[env_ids] = sampled
        self.task_mask[env_ids] = False
        self.task_mask[env_ids, sampled] = True

        for name, idx in self.task_name_to_id.items():
            mask = sampled == idx
            if not torch.any(mask):
                continue
            task_env_ids = env_ids[mask]
            if name == "sitdown":
                self._reset_sit_targets(task_env_ids)
            elif name == "carrybox":
                self._reset_carry_targets(task_env_ids)
            elif name == "standup":
                self._reset_stand_targets(task_env_ids)

    def _reset_sit_targets(self, env_ids):
        if env_ids.numel() == 0:
            return
        cfg = self._sit_cfg
        root_xy = self.root_states[env_ids, 0:2]
        distance = torch_rand_float(float(cfg.distance_range[0]), float(cfg.distance_range[1]), (env_ids.numel(), 1), device=self.device)
        angles = torch_rand_float(-torch.pi, torch.pi, (env_ids.numel(), 1), device=self.device)
        offsets_xy = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1) * distance
        offsets_xy += torch_rand_float(-float(cfg.lateral_noise), float(cfg.lateral_noise), (env_ids.numel(), 2), device=self.device)
        target_xy = root_xy + offsets_xy
        target_z = torch_rand_float(float(cfg.height_range[0]), float(cfg.height_range[1]), (env_ids.numel(), 1), device=self.device)

        self._sit_target_pos[env_ids, 0:2] = target_xy
        self._sit_target_pos[env_ids, 2:3] = target_z

        facing_angle = torch.atan2(-offsets_xy[:, 1], -offsets_xy[:, 0])
        facing_angle += torch_rand_float(-float(cfg.facing_noise), float(cfg.facing_noise), (env_ids.numel(),), device=self.device)
        self._sit_target_facing[env_ids, 0] = torch.cos(facing_angle)
        self._sit_target_facing[env_ids, 1] = torch.sin(facing_angle)
        self._sit_target_height[env_ids, 0] = target_z.squeeze(-1)
        sit_axis = torch.zeros(env_ids.numel(), 3, device=self.device)
        sit_axis[:, 2] = 1.0
        self._sit_target_quat[env_ids] = quat_from_angle_axis(facing_angle, sit_axis)

    def _reset_carry_targets(self, env_ids):
        if env_ids.numel() == 0:
            return
        cfg = self._carry_cfg
        root_pos = self.root_states[env_ids, 0:3]
        dist = torch_rand_float(float(cfg.start_distance_range[0]), float(cfg.start_distance_range[1]), (env_ids.numel(), 1), device=self.device)
        heading = torch_rand_float(-torch.pi, torch.pi, (env_ids.numel(), 1), device=self.device)
        dir_xy = torch.cat([torch.cos(heading), torch.sin(heading)], dim=-1)
        box_xy = root_pos[:, 0:2] + dir_xy * dist
        box_z = torch.full((env_ids.numel(), 1), float(cfg.height), device=self.device)
        self._carry_box_pos[env_ids, 0:2] = box_xy
        self._carry_box_pos[env_ids, 2:3] = box_z

        goal_dist = torch_rand_float(float(cfg.goal_distance_range[0]), float(cfg.goal_distance_range[1]), (env_ids.numel(), 1), device=self.device)
        goal_heading = heading + torch_rand_float(-0.5, 0.5, (env_ids.numel(), 1), device=self.device)
        goal_xy = box_xy + torch.cat([torch.cos(goal_heading), torch.sin(goal_heading)], dim=-1) * goal_dist
        self._carry_box_goal[env_ids, 0:2] = goal_xy
        self._carry_box_goal[env_ids, 2:3] = box_z
        carry_axis = torch.zeros(env_ids.numel(), 3, device=self.device)
        carry_axis[:, 2] = 1.0
        self._carry_box_rot[env_ids] = quat_from_angle_axis(heading.squeeze(-1), carry_axis)
        self._carry_box_start[env_ids] = self._carry_box_pos[env_ids]

    def _reset_stand_targets(self, env_ids):
        if env_ids.numel() == 0:
            return
        self._stand_target_height[env_ids, 0] = float(self._stand_cfg.target_height)
        marker_offset = torch.zeros((env_ids.numel(), 2), device=self.device)
        marker_offset[:, 0] = torch_rand_float(0.8, 1.5, (env_ids.numel(),), device=self.device)
        marker_offset[:, 1] = torch_rand_float(-0.5, 0.5, (env_ids.numel(),), device=self.device)
        self._stand_marker_pos[env_ids, 0:2] = self.root_states[env_ids, 0:2] + marker_offset
        self._stand_marker_pos[env_ids, 2] = self._stand_target_height[env_ids, 0]

    def _compute_sit_obs(self):
        delta = self._sit_target_pos - self.root_states[:, 0:3]
        facing = self._sit_target_facing
        height_error = self._sit_target_height - self.root_states[:, 2:3]
        return torch.cat([delta, facing, height_error], dim=-1)

    def _compute_carry_obs(self):
        delta_box = self._carry_box_pos - self.root_states[:, 0:3]
        delta_goal = self._carry_box_goal - self._carry_box_pos
        vel = self.base_lin_vel
        return torch.cat([delta_box, delta_goal, vel], dim=-1)

    def _compute_stand_obs(self):
        height = self.root_states[:, 2:3]
        height_error = self._stand_target_height - height
        up_dir = self.projected_gravity
        alignment = torch.sum(up_dir * self._stand_up_dir, dim=-1, keepdim=True)
        vel_z = self.base_lin_vel[:, 2:3]
        return torch.cat([height, height_error, up_dir[:, 2:3], vel_z, alignment], dim=-1)

    def _update_traj_markers(self):
        if not self.enable_traj_task:
            return
        S = self._num_traj_samples
        ars = self.all_root_states.view(self.num_envs, self.num_actors_per_env, 13)

        # x,y 来自轨迹；z 始终固定为每个 env 的 marker_z0
        zcol = self._marker_z0.view(self.num_envs, 1)  # (N,1)

        # 批量更新采样点
        ars[:, 1:1+S, 0:2] = self._traj_samples_buf[:, :, 0:2]
        ars[:, 1:1+S, 2]   = zcol.expand(-1, S)

        # 更新目标点
        ars[:, 1+S, 0:2] = self._traj_curr_target[:, 0:2]
        ars[:, 1+S, 2]   = self._marker_z0

        # 一次性写回
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.all_root_states.reshape(-1, 13)))

    # ---------------- Buffers ----------------
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.num_one_step_proprio_obs + 3, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = 0.  # command
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:(9 + self.num_dof)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(9 + self.num_dof):(9 + 2 * self.num_dof)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(9 + 2 * self.num_dof):(9 + 2 * self.num_dof + 15)] = noise_scales.end_effector * noise_level
        noise_vec[(9 + 2 * self.num_dof):(9 + 2 * self.num_dof + self.num_actions)] = 0.
        noise_vec[(9 + 2 * self.num_dof + self.num_actions + 15):(12 + 2 * self.num_dof + self.num_actions + 15)] = 0.
        return noise_vec

    def _init_buffers(self):
        # 获取 gym GPU 张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._refresh_sim_tensors()

        # root states（按 actor 组织）
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        assert self.all_root_states.numel() == self.num_envs * self.num_actors_per_env * 13
        ars = self.all_root_states.view(self.num_envs, self.num_actors_per_env, 13)
        self.root_states = ars[:, 0, :]  # 机器人在槽位 0

        # rigid bodies（只保留机器人刚体段）
        full_rb = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.total_bodies_per_env, 13)
        self.rigid_body_states = full_rb[:, :self.num_bodies, :]

        # dof / contact
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dv = self.dof_state.view(self.num_envs, self.num_dof, 2)
        self.dof_pos = dv[..., 0]
        self.dof_vel = dv[..., 1]
        full_cf = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.total_bodies_per_env, 3)
        self.contact_forces = full_cf[:, :self.num_bodies, :]

        # 常用缓存
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos  = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel  = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.left_feet_pos  = self.rigid_body_states[:, self.left_feet_indices, 0:3]
        self.right_feet_pos = self.rigid_body_states[:, self.right_feet_indices, 0:3]
        self.pelvis_contact_pos = self.rigid_body_states[:, self.pelvis_contact_index, :3]

        self.common_step_counter = 0
        self.extras = {}
        self.skill = self.cfg.asset.skill
        self.skill_init_prob = torch.tensor(self.cfg.asset.skill_init_prob, device=self.device)

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.z_axis_unit = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0)
        self.torques = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.p_gains = torch.zeros(self.num_dof, device=self.device)
        self.d_gains = torch.zeros(self.num_dof, device=self.device)

        # 明确 num_actions
        self.num_actions = getattr(self, "num_actions", self.num_dof)

        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.computed_torques = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, device=self.device)
        self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.first_contacts = torch.zeros_like(self.last_contacts)

        self.base_lin_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,7:10])
        self.base_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,10:13])
        self.projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.gravity_vec)

        self.delay_buffer = torch.zeros(self.cfg.domain_rand.max_delay_timesteps, self.num_envs, self.num_actions, device=self.device)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        self.end_effector_pos = torch.cat((
            self.rigid_body_states[:, self.hand_pos_indices[0], :3],
            self.rigid_body_states[:, self.hand_pos_indices[1], :3],
            self.feet_pos[:, 0], self.feet_pos[:, 1],
            self.rigid_body_states[:, self.head_index, :3]
        ), dim=-1)
        self.end_effector_pos = self.end_effector_pos - self.root_states[:, :3].repeat(1, 5)
        for i in range(5):
            self.end_effector_pos[:, 3*i:3*i+3] = quat_rotate_inverse(
                self.rigid_body_states[:, self.upper_body_index, 3:7],
                self.end_effector_pos[:, 3*i:3*i+3])

        if self.enable_traj_task:
            self.traj_obs_buf = torch.zeros(self.num_envs, self._task_obs_total_dim, device=self.device)
            self._traj_curr_target = torch.zeros(self.num_envs, 3, device=self.device)

        # 默认关节与 PD
        self.default_dof_pos = torch.zeros(self.num_dof, device=self.device)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness:
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found and self.cfg.control.control_type in ["P", "V"]:
                print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_poses = self.default_dof_pos.repeat(self.num_envs, 1)

        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, device=self.device)
        self.Kd_factors = torch.ones_like(self.Kp_factors)
        self.actuation_offset = torch.zeros_like(self.Kp_factors)
        self.motor_strength = torch.ones_like(self.Kp_factors)
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self.zero_force = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        # joint_powers 历史缓存（N_env, T_hist, N_dof）
        hist = max(int(self.actor_history_length), 1)
        self.joint_powers = torch.zeros(self.num_envs, hist, self.num_dof, device=self.device)

        actor_one_step_dim = ( 3                 # commands
                              +3                # base_ang_vel
                              +3                # gravity
                              +self.num_dof     # (q - q0)
                              +self.num_dof     # dq
                              +15               # ee pos
                              +self.num_actions # actions
                              +3 )              # base_lin_vel
        if self.enable_traj_task:
            actor_one_step_dim += self.num_task_obs
        self.obs_buf = torch.zeros(self.num_envs, self.actor_history_length * self.num_one_step_actor_obs, device=self.device)

        # privileged obs（不堆叠）
        priv_dim = ( 3+3+3 + self.num_dof + self.num_dof + 15 + self.num_actions + 3 )
        if self.enable_traj_task:
            priv_dim += self.num_task_obs
        self.privileged_obs_buf = torch.zeros(self.num_envs, priv_dim, device=self.device)

        # rew_buf 显式初始化
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)

        # dataset / AMP
        motion_file = self.cfg.dataset.motion_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        joint_mapping_file = self.cfg.dataset.joint_mapping_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.motionlib = MotionLib(
            motion_file=motion_file, mapping_file=joint_mapping_file, dof_names=self.dof_names,
            fps=self.cfg.dataset.frame_rate, device=self.device,
            window_length=self.cfg.amp.window_length, ratio_random_range=self.cfg.amp.ratio_random_range
        )
        amp_obs_joint_id = [i for i, n in enumerate(self.dof_names) if n in self.motionlib.mapping.keys()]
        self.amp_obs_joint_id = torch.tensor(amp_obs_joint_id, device=self.device)

    # ---------------- Rewards plumbing ----------------
    def _prepare_reward_function(self):
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    # ---------------- Terrain ----------------
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution
        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    # ---------------- Multi-actor creation (robot + markers) ----------------
    def _create_envs(self):
        # Robot asset
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        opt = gymapi.AssetOptions()
        opt.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        opt.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        opt.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        opt.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        opt.fix_base_link = self.cfg.asset.fix_base_link
        opt.density = self.cfg.asset.density
        opt.angular_damping = self.cfg.asset.angular_damping
        opt.linear_damping = self.cfg.asset.linear_damping
        opt.max_angular_velocity = self.cfg.asset.max_angular_velocity
        opt.max_linear_velocity = self.cfg.asset.max_linear_velocity
        opt.armature = self.cfg.asset.armature
        opt.thickness = self.cfg.asset.thickness
        opt.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, opt)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dof = len(self.dof_names)

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        left_foot_names = [s for s in body_names if self.cfg.asset.left_foot_name in s]
        right_foot_names = [s for s in body_names if self.cfg.asset.right_foot_name in s]

        penalized_contact_names, termination_contact_names = [], []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names += [s for s in body_names if name in s]
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names += [s for s in body_names if name in s]

        hand_pos_names  = [s for s in body_names if self.cfg.asset.hand_pos_name  in s]
        hand_colli_names = [s for s in body_names if self.cfg.asset.hand_colli_name in s]
        self.torso_link_index = body_names.index("torso_link")

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

        # Marker asset（仅当启用轨迹任务）
        self.num_markers_per_env = (self._num_traj_samples + 1) if self.enable_traj_task else 0
        marker_asset = None
        self.marker_num_bodies = 0
        if self.enable_traj_task:
            m_path = self.cfg.marker.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            m_root, m_file = os.path.dirname(m_path), os.path.basename(m_path)
            mopt = gymapi.AssetOptions()
            mopt.angular_damping = 0.01
            mopt.linear_damping = 0.01
            mopt.max_angular_velocity = 100.0
            mopt.density = 1.0
            mopt.fix_base_link = True
            mopt.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            marker_asset = self.gym.load_asset(self.sim, m_root, m_file, mopt)
            self.marker_num_bodies = self.gym.get_asset_rigid_body_count(marker_asset)

        # 多 actor 规模
        self.num_actors_per_env = 1 + self.num_markers_per_env
        self.total_bodies_per_env = int(self.num_bodies + (self.marker_num_bodies * self.num_markers_per_env))

        # 创建 env
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.envs, self.actor_handles = [], []
        if self.enable_traj_task:
            self.marker_handles = [[] for _ in range(self.num_envs)]
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device)

        # 准备 marker_z0 向量
        self._marker_z0 = torch.zeros(self.num_envs, device=self.device)

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env)

            # 机器人（槽位 0）
            start_pose = gymapi.Transform()
            pos = (self.env_origins[i] + self.base_init_state[:3]).tolist()
            start_pose.p = gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            robot = self.gym.create_actor(env, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env, robot, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env, robot)

            if i == 0:
                self.default_com_torso = copy.deepcopy(body_props[self.torso_link_index].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env, robot, body_props, recomputeInertia=True)
            self.actor_handles.append(robot)

            # markers（槽位 1..S+1）
            if self.enable_traj_task:
                # 第一帧与机器人初始化位置对齐（x,y），z 固定 = base_init_z + height_offset
                base_z0 = float(pos[2] + self._marker_height_offset)
                self._marker_z0[i] = base_z0

                def _finalize_marker(handle):
                    # 颜色：红色
                    self.gym.set_rigid_body_color(env, handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.0, 0.0))
                    # 关闭一切碰撞
                    shape_props = self.gym.get_actor_rigid_shape_properties(env, handle)
                    for j in range(len(shape_props)):
                        shape_props[j].filter = 0xFFFF
                    self.gym.set_actor_rigid_shape_properties(env, handle, shape_props)

                for k in range(self._num_traj_samples):  # 采样点
                    mpose = gymapi.Transform()
                    mpose.p = gymapi.Vec3(float(pos[0]), float(pos[1]), base_z0)
                    h = self.gym.create_actor(env, marker_asset, mpose, f"{self.cfg.marker.asset.name}_sample_{k}", i, 0, 0)
                    _finalize_marker(h)
                    self.marker_handles[i].append(h)

                # 目标点
                tpose = gymapi.Transform()
                tpose.p = gymapi.Vec3(float(pos[0]), float(pos[1]), base_z0)
                h = self.gym.create_actor(env, marker_asset, tpose, f"{self.cfg.marker.asset.name}_target", i, 0, 0)
                _finalize_marker(h)
                self.marker_handles[i].append(h)

        # 常用索引
        self.left_hip_joint_indices  = torch.tensor([self.dof_names.index(n) for n in self.cfg.control.left_hip_joints],  dtype=torch.long, device=self.device)
        self.right_hip_joint_indices = torch.tensor([self.dof_names.index(n) for n in self.cfg.control.right_hip_joints], dtype=torch.long, device=self.device)
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))

        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.tensor([self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in knee_names], dtype=torch.long, device=self.device)

        hand_pos_indices = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in hand_pos_names]
        self.hand_pos_indices = torch.tensor(hand_pos_indices, dtype=torch.long, device=self.device)
        hand_colli_indices = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in hand_colli_names]
        self.hand_colli_indices = torch.tensor(hand_colli_indices, dtype=torch.long, device=self.device)
        self.head_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.head_name)
        self.pelvis_contact_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], self.cfg.asset.pelvis_contact_name
        )

        self.feet_indices = torch.tensor([self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in feet_names], dtype=torch.long, device=self.device)
        self.left_feet_indices  = torch.tensor([self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in left_foot_names], dtype=torch.long, device=self.device)
        self.right_feet_indices = torch.tensor([self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in right_foot_names], dtype=torch.long, device=self.device)

        self.penalised_contact_indices = torch.tensor([self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in penalized_contact_names], dtype=torch.long, device=self.device)
        self.termination_contact_indices = torch.tensor([self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in termination_contact_names], dtype=torch.long, device=self.device)

        self.left_leg_joint_indices  = torch.tensor([self.dof_names.index(n) for n in self.cfg.control.left_leg_joints],  dtype=torch.long, device=self.device)
        self.right_leg_joint_indices = torch.tensor([self.dof_names.index(n) for n in self.cfg.control.right_leg_joints], dtype=torch.long, device=self.device)
        self.leg_joint_indices = torch.cat((self.left_leg_joint_indices, self.right_leg_joint_indices))

        self.left_arm_joint_indices  = torch.tensor([self.dof_names.index(n) for n in self.cfg.control.left_arm_joints],  dtype=torch.long, device=self.device)
        self.right_arm_joint_indices = torch.tensor([self.dof_names.index(n) for n in self.cfg.control.right_arm_joints], dtype=torch.long, device=self.device)
        self.arm_joint_indices = torch.cat((self.left_arm_joint_indices, self.right_arm_joint_indices))

        self.waist_joint_indices = torch.tensor([self.dof_names.index(n) for n in self.cfg.asset.waist_joints], dtype=torch.long, device=self.device)
        self.ankle_joint_indices = torch.tensor([self.dof_names.index(n) for n in self.cfg.asset.ankle_joints], dtype=torch.long, device=self.device)
        self.upper_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.control.upper_body_link)

        self.keyframe_names = [s for s in body_names if self.cfg.asset.keyframe_name in s]
        self.keyframe_indices = torch.tensor(
            [self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n) for n in self.keyframe_names],
            dtype=torch.long, device=self.device)

        # actor id 映射（按创建顺序：0=robot, 1..S+1=markers）
        base_ids = (self.num_actors_per_env * torch.arange(self.num_envs, device=self.device, dtype=torch.int32))
        self._robot_actor_ids  = base_ids + 0
        self._marker_actor_ids = []
        for k in range(self.num_markers_per_env):
            self._marker_actor_ids.append(base_ids + (k + 1))

    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _get_base_heights(self, env_ids=None):
        return self.root_states[:, 2].clone()

    # ------------ rewards ------------
    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_vel(self):
        lin_vel_error = torch.sum(torch.square(self.cfg.rewards.target_vel * self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_traj_tracking(self):
        if not self.enable_traj_task:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        root_pos = self.root_states[:, 0:3]
        return compute_traj_reward(root_pos, self._traj_curr_target)

    def _reward_loco_task(self):
        sit_idx = self.task_name_to_id.get("sitdown", None)
        if sit_idx is None:
            return torch.zeros(self.num_envs, device=self.device)
        mask = self.task_mask[:, sit_idx]
        if not torch.any(mask):
            return torch.zeros(self.num_envs, device=self.device)

        chair_pos = self._sit_target_pos
        robot2chair_dir = chair_pos[:, :2] - self.root_states[:, :2]
        robot2chair_dist_xy = torch.norm(robot2chair_dir, dim=-1)
        robot2chair_dir = normalize(robot2chair_dir)
        global_lin_vel = self.rigid_body_states[:, self.upper_body_index, 7:10]
        robot2chair_vel = torch.sum(robot2chair_dir * global_lin_vel[:, :2], dim=-1)
        robot2chair_vel_reward = torch.exp(-5 * torch.square(self.cfg.rewards.target_speed_loco - robot2chair_vel))

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        target_heading = torch.atan2(
            chair_pos[:, 1] - self.root_states[:, 1],
            chair_pos[:, 0] - self.root_states[:, 0],
        )
        yaw_error = torch.abs(wrap_to_pi(target_heading - heading))
        loco_heading_reward = torch.exp(-0.75 * yaw_error)

        loco_reward = (
            self.cfg.rewards.robot2chair_vel * robot2chair_vel_reward
            + self.cfg.rewards.loco_heading * loco_heading_reward
        )

        thresh = float(self.cfg.rewards.thresh_robot2chair)
        loco_reward[robot2chair_dist_xy < thresh] = (
            self.cfg.rewards.robot2chair_vel + self.cfg.rewards.loco_heading
        )

        return loco_reward * mask.float()

    def _reward_sitDown_task(self):
        sit_idx = self.task_name_to_id.get("sitdown", None)
        if sit_idx is None:
            return torch.zeros(self.num_envs, device=self.device)
        mask = self.task_mask[:, sit_idx]
        if not torch.any(mask):
            return torch.zeros(self.num_envs, device=self.device)

        chair_pos = self._sit_target_pos
        chair_quat = self._sit_target_quat
        robot2chair_dir = chair_pos[:, :2] - self.root_states[:, :2]
        robot2chair_dist_xy = torch.norm(robot2chair_dir, dim=-1)

        sit_pos_far_reward = 1.0 - robot2chair_dist_xy / float(self.cfg.rewards.thresh_robot2chair)
        sit_pos_far_reward = torch.clamp(sit_pos_far_reward, min=0.0)
        sit_pos_far_reward[robot2chair_dist_xy < 0.3] = 1.0

        contact2chair_dist = torch.norm(self.pelvis_contact_pos - chair_pos, dim=-1)
        sit_pos_near_reward = torch.exp(-3 * contact2chair_dist)

        sit_height_error = torch.abs(chair_pos[:, 2] - self.pelvis_contact_pos[:, 2])
        sit_height_reward = torch.exp(-5.0 * sit_height_error)
        sit_height_reward[robot2chair_dist_xy > 0.3] = 0.0

        robot_forward = quat_apply(self.base_quat, self.forward_vec)
        robot_heading = torch.atan2(robot_forward[:, 1], robot_forward[:, 0])
        chair_forward = quat_apply(chair_quat, self.forward_vec)
        chair_heading = torch.atan2(chair_forward[:, 1], chair_forward[:, 0])
        yaw_error = torch.abs(wrap_to_pi(robot_heading - chair_heading))
        sit_heading_reward = torch.exp(-0.75 * yaw_error)

        sit_reward = (
            self.cfg.rewards.sit_pos_far * sit_pos_far_reward
            + self.cfg.rewards.sit_pos_near * sit_pos_near_reward
            + self.cfg.rewards.sit_height * sit_height_reward
            + self.cfg.rewards.sit_heading * sit_heading_reward
        )
        sit_reward[robot2chair_dist_xy > float(self.cfg.rewards.thresh_robot2chair)] = 0.0

        return sit_reward * mask.float()

    def _reward_walk_task(self):
        carry_idx = self.task_name_to_id.get("carrybox", None)
        if carry_idx is None:
            return torch.zeros(self.num_envs, device=self.device)
        mask = self.task_mask[:, carry_idx]
        if not torch.any(mask):
            return torch.zeros(self.num_envs, device=self.device)

        box_pos = self._carry_box_pos
        robot_pos = self.root_states[:, :3]
        robot2object_vec = box_pos[:, :2] - robot_pos[:, :2]
        robot2object_dist = torch.norm(robot2object_vec, dim=-1)
        robot2object_dir = normalize(robot2object_vec)

        global_lin_vel = self.rigid_body_states[:, self.upper_body_index, 7:10]
        robot2object_vel = torch.sum(robot2object_dir * global_lin_vel[:, :2], dim=-1)
        robot2object_vel_reward = torch.exp(-5 * torch.square(self.cfg.rewards.target_speed_loco - robot2object_vel))

        robot_forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(robot_forward[:, 1], robot_forward[:, 0])
        target_heading = torch.atan2(
            box_pos[:, 1] - robot_pos[:, 1], box_pos[:, 0] - robot_pos[:, 0]
        )
        yaw_error = torch.abs(wrap_to_pi(target_heading - heading))
        start_heading_reward = torch.exp(-0.75 * yaw_error)

        robot2object_pos_reward = torch.exp(-0.5 * robot2object_dist)

        walk_reward = (
            self.cfg.rewards.robot2object_pos * robot2object_pos_reward
            + self.cfg.rewards.robot2object_vel * robot2object_vel_reward
            + self.cfg.rewards.start_heading * start_heading_reward
        )

        thresh_object = float(self.cfg.rewards.thresh_robot2object)
        walk_reward[robot2object_dist < thresh_object] = (
            self.cfg.rewards.robot2object_pos
            + self.cfg.rewards.robot2object_vel
            + self.cfg.rewards.start_heading
        )

        object2goal_dist = torch.norm(self._carry_box_goal - box_pos, dim=-1)
        walk_reward[object2goal_dist < float(self.cfg.rewards.thresh_object2goal)] = (
            self.cfg.rewards.robot2object_pos
            + self.cfg.rewards.robot2object_vel
            + self.cfg.rewards.start_heading
        )

        return walk_reward * mask.float()

    def _reward_carryup_task(self):
        carry_idx = self.task_name_to_id.get("carrybox", None)
        if carry_idx is None:
            return torch.zeros(self.num_envs, device=self.device)
        mask = self.task_mask[:, carry_idx]
        if not torch.any(mask):
            return torch.zeros(self.num_envs, device=self.device)

        box_pos = self._carry_box_pos
        goal_pos = self._carry_box_goal
        hand_pos = self.rigid_body_states[:, self.hand_pos_indices, :3]
        hand2object_err = torch.sum((hand_pos.mean(dim=1) - box_pos) ** 2, dim=-1)
        hand2object_reward = torch.exp(-3 * hand2object_err)

        box_carryup_reward = torch.exp(
            -3 * torch.clamp(self.cfg.rewards.target_box_height - box_pos[:, 2], min=0)
        )
        box_carryup_reward[box_pos[:, 2] > self.cfg.rewards.target_box_height] = 1.0
        object2goal_dist = torch.norm(goal_pos - box_pos, dim=-1)
        box_carryup_reward[object2goal_dist < float(self.cfg.rewards.thresh_object2goal)] = 1.0

        carryup_reward = (
            self.cfg.rewards.hand_pos * hand2object_reward
            + self.cfg.rewards.box_height * box_carryup_reward
        )

        robot2object_dist = torch.norm(box_pos[:, :2] - self.root_states[:, :2], dim=-1)
        carryup_reward[robot2object_dist > float(self.cfg.rewards.thresh_robot2object)] = 0.0
        carryup_reward[object2goal_dist < float(self.cfg.rewards.thresh_object2goal)] = (
            self.cfg.rewards.hand_pos + self.cfg.rewards.box_height
        )

        return carryup_reward * mask.float()

    def _reward_relocation_task(self):
        carry_idx = self.task_name_to_id.get("carrybox", None)
        if carry_idx is None:
            return torch.zeros(self.num_envs, device=self.device)
        mask = self.task_mask[:, carry_idx]
        if not torch.any(mask):
            return torch.zeros(self.num_envs, device=self.device)

        box_pos = self._carry_box_pos
        goal_pos = self._carry_box_goal
        robot_pos = self.root_states[:, :3]

        robot_forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(robot_forward[:, 1], robot_forward[:, 0])
        target_heading = torch.atan2(
            goal_pos[:, 1] - robot_pos[:, 1], goal_pos[:, 0] - robot_pos[:, 0]
        )
        yaw_error = torch.abs(wrap_to_pi(target_heading - heading))
        relocation_heading_reward = torch.exp(-0.75 * yaw_error)

        heading_error = 0.5 * wrap_to_pi(target_heading - heading)
        ang_command = torch.clip(heading_error, -1.0, 1.0)
        ang_vel_error = torch.square(ang_command - self.base_ang_vel[:, 2])
        relocation_heading_vel_reward = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

        robot2goal_vec = goal_pos[:, :2] - robot_pos[:, :2]
        robot2goal_dir = normalize(robot2goal_vec)
        robot2goal_dist = torch.norm(robot2goal_vec, dim=-1)
        robot2goal_pos_reward = torch.exp(-0.5 * robot2goal_dist)
        global_lin_vel = self.rigid_body_states[:, self.upper_body_index, 7:10]
        robot2goal_vel = torch.sum(robot2goal_dir * global_lin_vel[:, :2], dim=-1)
        robot2goal_vel_reward = torch.exp(-5 * torch.square(self.cfg.rewards.target_speed_carry - robot2goal_vel))

        object2goal_vec = goal_pos - box_pos
        object2goal_dist = torch.norm(object2goal_vec, dim=-1)
        object2goal_pos_reward = torch.exp(-10.0 * object2goal_dist)

        robot2goal_pos_reward[robot2goal_dist < float(self.cfg.rewards.thresh_robot2goal)] = 1.0
        robot2goal_vel_reward[robot2goal_dist < float(self.cfg.rewards.thresh_robot2goal)] = 1.0

        put_box_reward = torch.exp(-3.0 * torch.abs(box_pos[:, 2] - goal_pos[:, 2]))
        object2goal_dist_xy = torch.norm(object2goal_vec[:, :2], dim=-1)
        put_box_reward[object2goal_dist_xy > 0.6] = 0.0

        relocation_reward = (
            self.cfg.rewards.relocation_heading * relocation_heading_reward
            + self.cfg.rewards.relocation_heading_vel * relocation_heading_vel_reward
            + self.cfg.rewards.robot2goal_pos * robot2goal_pos_reward
            + self.cfg.rewards.robot2goal_vel * robot2goal_vel_reward
            + self.cfg.rewards.object2goal_pos * object2goal_pos_reward
            + self.cfg.rewards.put_box * put_box_reward
        )

        lift_height = box_pos[:, 2] - self._carry_box_start[:, 2]
        is_stage_relocation = (lift_height > 0.05) | (
            torch.norm(box_pos[:, :2] - self._carry_box_start[:, :2], dim=-1)
            > float(self.cfg.rewards.thresh_object2start)
        )
        relocation_reward[~is_stage_relocation] = 0.0
        relocation_reward[object2goal_dist < float(self.cfg.rewards.thresh_object2goal)] = (
            self.cfg.rewards.relocation_heading
            + self.cfg.rewards.relocation_heading_vel
            + self.cfg.rewards.robot2goal_pos
            + self.cfg.rewards.robot2goal_vel
            + self.cfg.rewards.object2goal_pos
            + self.cfg.rewards.put_box
        )

        return relocation_reward * mask.float()

    def _reward_standup_task(self):
        stand_idx = self.task_name_to_id.get("standup", None)
        if stand_idx is None:
            return torch.zeros(self.num_envs, device=self.device)
        mask = self.task_mask[:, stand_idx]
        if not torch.any(mask):
            return torch.zeros(self.num_envs, device=self.device)

        base_height = self.root_states[:, 2]
        height_error = torch.clamp(self._stand_target_height[:, 0] - base_height, min=0.0)
        stand_reward = torch.exp(-5.0 * height_error)
        stand_reward[height_error <= 0.0] = 1.0

        head_height = self.rigid_body_states[:, self.head_index, 2]
        head_height_error = torch.abs(head_height - (self._stand_target_height[:, 0] + 0.2))
        head_reward = torch.exp(-2 * head_height_error)
        head_reward[head_height > self._stand_target_height[:, 0] + 0.2] = 1.0

        stand_reward = (
            self.cfg.rewards.base_height * stand_reward
            + self.cfg.rewards.head_height * head_reward
        )

        robot2marker = torch.norm(self._stand_marker_pos[:, :2] - self.root_states[:, :2], dim=-1)
        near_reward = torch.exp(-2 * robot2marker)
        stand_reward += (
            self.cfg.rewards.stand_still
            * torch.exp(-0.3 * torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1))
        )
        stand_reward += self.cfg.rewards.hand_free * torch.mean(
            1.0 * (torch.norm(self.contact_forces[:, self.hand_colli_indices], dim=-1) < 1.0), dim=1
        )
        stand_reward += self.cfg.rewards.pos_near * near_reward

        return stand_reward * mask.float()

    def _reward_tracking_yaw(self):
        rew = torch.exp(-torch.abs(self.commands[:,2] - self.yaw[:,0]))
        return rew

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_joint_power(self):
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1) / torch.clip(torch.sum(torch.square(self.commands[:,0:1]), dim=-1), min=0.01)

    def _reward_base_height(self):
        base_height = self._get_base_heights()
        return torch.abs(base_height - self.cfg.rewards.base_height_target)

    def _reward_base_height_wrt_feet(self):
        base_height_l = self.root_states[:, 2] - self.feet_pos[:, 0, 2]
        base_height_r = self.root_states[:, 2] - self.feet_pos[:, 1, 2]
        base_height = torch.max(base_height_l, base_height_r)
        return torch.abs(base_height - self.cfg.rewards.base_height_target)

    def _reward_feet_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_smoothness(self):
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques / self.p_gains.unsqueeze(0)), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_success_termination(self):
        return self.success_buf

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        return torch.sum((torch.abs(self.computed_torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_feet_air_time(self):
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * self.first_contacts, dim=1)
        rew_airTime *= torch.norm(self.commands[:, 0:1], dim=1) > 0.1
        return rew_airTime

    def _reward_feet_stumble(self):
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 3 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, 0:2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        rew_no_fly = 1.0 * single_contact
        rew_no_fly = torch.max(rew_no_fly, 1. * (torch.norm(self.commands[:, 0:1], dim=1) < 0.1))
        return rew_no_fly

    def _reward_joint_tracking_error(self):
        return torch.sum(torch.square(self.joint_pos_target - self.dof_pos), dim=-1)

    def _reward_joint_deviation(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew

    def _reward_arm_joint_deviation(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[self.arm_joint_indices], dim=-1)

    def _reward_leg_joint_deviation(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[self.leg_joint_indices], dim=-1)

    def _reward_leg_power_symmetry(self):
        left_leg_power = torch.mean(self.joint_powers[:, :, self.left_leg_joint_indices], dim=1)
        right_leg_power = torch.mean(self.joint_powers[:, :, self.right_leg_joint_indices], dim=1)
        leg_power_diff = torch.abs(left_leg_power - right_leg_power).mean(dim=1)
        return leg_power_diff

    def _reward_arm_power_symmetry(self):
        left_arm_power = torch.sum(self.joint_powers[:, :, self.left_arm_joint_indices], dim=1)
        right_arm_power = torch.sum(self.joint_powers[:, :, self.right_arm_joint_indices], dim=1)
        arm_power_diff = torch.abs(left_arm_power - right_arm_power).mean(dim=1)
        return arm_power_diff

    def _reward_feet_distance_lateral(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        foot_leteral_dis = torch.abs(footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1])
        return torch.clip(foot_leteral_dis - self.cfg.rewards.least_feet_distance_lateral, max=0) + torch.clip(self.cfg.rewards.max_feet_distance_lateral - foot_leteral_dis, max=0)

    def _reward_feet_ground_parallel(self):
        left_height_std = torch.std(self.left_feet_pos[:, :, 2], dim=1).view(-1, 1)
        right_height_std = torch.std(self.right_feet_pos[:, :, 2], dim=1).view(-1, 1)
        return torch.sum(torch.cat((left_height_std, right_height_std), dim=1) * self.contact_filt, dim=-1)

    def _reward_feet_parallel(self):
        feet_distances = torch.norm(self.left_feet_pos[:, :, :2] - self.right_feet_pos[:, :, :2], dim=-1)
        return torch.std(feet_distances, dim=-1)

    def _reward_knee_distance_lateral(self):
        cur_knee_pos_translated = self.rigid_body_states[:, self.knee_indices, :3].clone() - self.root_states[:, 0:3].unsqueeze(1)
        knee_pos_in_body_frame = torch.zeros(self.num_envs, len(self.knee_indices), 3, device=self.device)
        for i in range(len(self.knee_indices)):
            knee_pos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_knee_pos_translated[:, i, :])
        knee_lateral_dis = torch.abs(knee_pos_in_body_frame[:, 0, 1] - self.cfg.rewards.least_knee_distance_lateral)
        return torch.clamp(knee_lateral_dis - self.cfg.rewards.least_knee_distance_lateral, max=0)


# -------- helpers --------
def compute_location_observations(root_states, traj_samples):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = calc_heading_quat_inv(root_rot)
    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(1), (traj_samples.shape[0], traj_samples.shape[1], 4))
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (traj_samples.shape[0], traj_samples.shape[1], 3))
    local_traj_samples = quat_rotate(heading_rot_exp.reshape(-1, 4), traj_samples.reshape(-1, 3) - root_pos_exp.reshape(-1, 3))
    obs = local_traj_samples[..., 0:2].reshape(root_pos.shape[0], -1)  # (N_env, 2*S)
    return obs


@torch.jit.script
def compute_traj_reward(root_pos, tar_pos):
    pos_err_scale = 2.0
    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)
    reward = pos_reward
    return reward