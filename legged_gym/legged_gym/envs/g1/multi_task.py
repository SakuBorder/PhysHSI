# SPDX-FileCopyrightText: Copyright (c) 2024 PhysHSI Authors
# SPDX-License-Identifier: BSD-3-Clause

import torch
from enum import Enum

from legged_gym.envs.g1.traj import LeggedRobot as G1TrajEnv
from legged_gym.utils.torch_utils import torch_rand_float


class G1MultiTask(G1TrajEnv):
    class TaskUID(Enum):
        traj = 0
        sitdown = 1
        carrybox = 2
        standup = 3

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._cfg_multi = cfg.multi_task
        self._task_names = list(self._cfg_multi.task_names)
        self._task_prob = torch.tensor(self._cfg_multi.task_init_prob, dtype=torch.float32)
        self._task_obs_dim = dict(self._cfg_multi.task_obs_dim)

        self._num_tasks = len(self._task_names)
        assert self._num_tasks == len(G1MultiTask.TaskUID)

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self._task_prob = self._task_prob.to(self.device)
        self._task_prob = self._task_prob / torch.clamp(self._task_prob.sum(), min=1e-6)
        self._task_indicator = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._task_mask = torch.zeros(self.num_envs, self._num_tasks, dtype=torch.bool, device=self.device)

        self._sit_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._sit_target_facing = torch.zeros(self.num_envs, 2, device=self.device)
        self._sit_target_height = torch.zeros(self.num_envs, 1, device=self.device)

        self._carry_box_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._carry_box_goal = torch.zeros(self.num_envs, 3, device=self.device)

        self._stand_target_height = torch.full((self.num_envs, 1), self._cfg_multi.standup.target_height, device=self.device)
        self._stand_up_dir = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(self.num_envs, 1)

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_multi_task(env_ids)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        self._reset_multi_task(env_ids)

    def compute_observations(self):
        super().compute_observations()
        self.extras["task_mask"] = self._task_mask
        self.extras["task_uid"] = self._task_indicator

    def compute_task_observations(self):
        traj_actor, _ = super().compute_task_observations()

        sit_obs = self._compute_sit_obs()
        carry_obs = self._compute_carry_obs()
        stand_obs = self._compute_stand_obs()

        obs_chunks = [traj_actor, sit_obs, carry_obs, stand_obs]
        full_obs = torch.cat(obs_chunks, dim=-1)
        mask_float = self._task_mask.float()
        full_obs = torch.cat([full_obs, mask_float], dim=-1)
        return full_obs, full_obs

    def _reset_multi_task(self, env_ids):
        if len(env_ids) == 0:
            return

        sampled_task = torch.multinomial(self._task_prob, len(env_ids), replacement=True)
        self._task_indicator[env_ids] = sampled_task
        self._task_mask[env_ids] = False
        self._task_mask[env_ids, sampled_task] = True

        for task_uid in G1MultiTask.TaskUID:
            mask = self._task_indicator[env_ids] == task_uid.value
            if not torch.any(mask):
                continue
            task_env_ids = env_ids[mask]
            if task_uid == G1MultiTask.TaskUID.traj:
                continue
            if task_uid == G1MultiTask.TaskUID.sitdown:
                self._reset_sit_targets(task_env_ids)
            elif task_uid == G1MultiTask.TaskUID.carrybox:
                self._reset_carry_targets(task_env_ids)
            elif task_uid == G1MultiTask.TaskUID.standup:
                self._reset_stand_targets(task_env_ids)

    def _reset_sit_targets(self, env_ids):
        cfg = self._cfg_multi.sitdown
        root_xy = self.root_states[env_ids, 0:2]
        distance = torch_rand_float(cfg.distance_range[0], cfg.distance_range[1], (len(env_ids), 1), device=self.device)
        angles = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device)
        offsets_xy = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1) * distance
        offsets_xy += torch_rand_float(-cfg.lateral_noise, cfg.lateral_noise, (len(env_ids), 2), device=self.device)
        target_xy = root_xy + offsets_xy
        target_z = torch_rand_float(cfg.height_range[0], cfg.height_range[1], (len(env_ids), 1), device=self.device)

        self._sit_target_pos[env_ids, 0:2] = target_xy
        self._sit_target_pos[env_ids, 2:3] = target_z

        facing_angle = torch.atan2(-offsets_xy[:, 1], -offsets_xy[:, 0])
        facing_angle += torch_rand_float(-cfg.facing_noise, cfg.facing_noise, (len(env_ids),), device=self.device)
        self._sit_target_facing[env_ids, 0] = torch.cos(facing_angle)
        self._sit_target_facing[env_ids, 1] = torch.sin(facing_angle)
        self._sit_target_height[env_ids, 0] = target_z.squeeze(-1)

    def _reset_carry_targets(self, env_ids):
        cfg = self._cfg_multi.carrybox
        root_pos = self.root_states[env_ids, 0:3]
        dist = torch_rand_float(cfg.start_distance_range[0], cfg.start_distance_range[1], (len(env_ids), 1), device=self.device)
        heading = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device)
        dir_xy = torch.cat([torch.cos(heading), torch.sin(heading)], dim=-1)
        box_xy = root_pos[:, 0:2] + dir_xy * dist
        box_z = torch.full((len(env_ids), 1), cfg.height, device=self.device)
        self._carry_box_pos[env_ids, 0:2] = box_xy
        self._carry_box_pos[env_ids, 2:3] = box_z

        goal_dist = torch_rand_float(cfg.goal_distance_range[0], cfg.goal_distance_range[1], (len(env_ids), 1), device=self.device)
        goal_heading = heading + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device)
        goal_xy = box_xy + torch.cat([torch.cos(goal_heading), torch.sin(goal_heading)], dim=-1) * goal_dist
        self._carry_box_goal[env_ids, 0:2] = goal_xy
        self._carry_box_goal[env_ids, 2:3] = box_z

    def _reset_stand_targets(self, env_ids):
        self._stand_target_height[env_ids, 0] = self._cfg_multi.standup.target_height

    def _compute_sit_obs(self):
        delta = self._sit_target_pos - self.root_states[:, 0:3]
        facing = self._sit_target_facing
        height_error = self._sit_target_height - self.root_states[:, 2:3]
        obs = torch.cat([delta, facing, height_error], dim=-1)
        return obs

    def _compute_carry_obs(self):
        delta_box = self._carry_box_pos - self.root_states[:, 0:3]
        delta_goal = self._carry_box_goal - self._carry_box_pos
        vel = self.base_lin_vel
        obs = torch.cat([delta_box, delta_goal, vel], dim=-1)
        return obs

    def _compute_stand_obs(self):
        height = self.root_states[:, 2:3]
        height_error = self._stand_target_height - height
        up_dir = self.projected_gravity
        alignment = torch.sum(up_dir * self._stand_up_dir, dim=-1, keepdim=True)
        vel_z = self.base_lin_vel[:, 2:3]
        obs = torch.cat([height, height_error, up_dir[:, 2:3], vel_z, alignment], dim=-1)
        return obs

