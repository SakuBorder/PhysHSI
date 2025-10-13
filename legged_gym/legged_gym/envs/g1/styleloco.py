# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import time
import copy
import numpy as np

import torch
import torch.nn.functional as F

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.torch_utils import calc_heading_quat_inv, quat_to_tan_norm, euler_from_quaternion

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.motionlib.motionlib_styleloco import MotionLib

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        
        self.num_one_step_proprio_obs = self.cfg.env.num_one_step_proprio_obs
        self.actor_history_length = self.cfg.env.num_actor_history
        self.actor_obs_length = self.cfg.env.num_actor_obs
        
        self.num_privileged_obs = self.cfg.env.num_privileged_obs

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.num_amp_obs = cfg.amp.num_obs
        self.init_done = True
        self.amp_obs_buf = torch.zeros(self.num_envs, self.num_amp_obs, device=self.device, dtype=torch.float)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs, amp_obs_buf = self.post_physics_step()
        
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs, amp_obs_buf

    def play_dataset_step(self, time):
        
        time = time % self.motionlib.motion_base_pos.shape[0]
        for env_id, env_ptr in enumerate(self.envs):
            self.root_states[env_id, 0:3] = self.motionlib.motion_base_pos[time]
            self.root_states[env_id, 3:7] = self.motionlib.motion_base_quat[time]
            self.root_states[env_id, 7:10] = self.motionlib.motion_global_lin_vel[time]
            self.root_states[env_id, 10:13] = self.motionlib.motion_global_ang_vel[time]
            self.dof_pos[env_id] = self.motionlib.motion_dof_pos[time]
            self.dof_vel[env_id] = self.motionlib.motion_dof_vel[time]

        env_ids_int32 = torch.arange(self.num_envs, device=self.device).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self._refresh_sim_tensors()
        self.render()
        self.common_step_counter += 1
        self.gym.simulate(self.sim)

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

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
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

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

        self.left_feet_pos = self.rigid_body_states[:, self.left_feet_indices, 0:3]
        self.right_feet_pos = self.rigid_body_states[:, self.right_feet_indices, 0:3]
        
        # compute contact related quantities
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) * self.contact_filt
        self.feet_air_time += self.dt
        
        # compute joint powers
        joint_powers = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((joint_powers, self.joint_powers[:, :-1]), dim=1)

        self._post_physics_step_callback()

        self.check_termination()

        self.compute_reward()

        amp_obs_buf = self.compute_amp_observations()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.disturbance[:, :, :] = 0.0
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        
        # reset contact related quantities
        self.feet_air_time *= ~self.contact_filt

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs, amp_obs_buf

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        
        self.reset_buf |= torch.logical_or(torch.abs(self.roll)>0.5, torch.abs(self.pitch-0.25)>0.85)
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.root_states[:, 2] < 0.35

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        if self.cfg.env.action_curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_action_curriculum(env_ids)

        self._reset_actors(env_ids)
        self._resample_commands(env_ids)
        self._reset_env_tensors(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.joint_powers[env_ids] = 0.
        self.delay_buffer[:, env_ids, :] = self.dof_pos[env_ids] - self.default_dof_pos
        self.reset_buf[env_ids] = 1
        
        # reset randomized prop
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

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.action_curriculum:
            self.extras["episode"]["action_curriculum_ratio"] = self.action_curriculum_ratio
        self.episode_length_buf[env_ids] = 0

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
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
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
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

        # add noise if needed
        if self.add_noise:
            current_actor_obs = current_actor_obs + (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[0:(9 + 2 * self.num_dof + self.num_actions + 15)]
        
        # actor & critic observations
        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_proprio_obs:], current_actor_obs), dim=-1)
        self.privileged_obs_buf = current_obs.clone()
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        # proprioceptive observations
        current_obs = torch.cat((self.commands[:, :3],
                                 self.base_ang_vel  * self.obs_scales.ang_vel,
                                 self.projected_gravity,
                                 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                 self.dof_vel * self.obs_scales.dof_vel,
                                 self.end_effector_pos,
                                 self.actions,
                                 self.base_lin_vel * self.obs_scales.lin_vel,
                                 ), dim=-1)
            
        return current_obs[env_ids]
        
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        start = time.time()
        print("*"*80)
        print("Start creating ground...")
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

        
    def create_cameras(self):
        """ Creates camera for each robot
        """
        self.camera_params = gymapi.CameraProperties()
        self.camera_params.width = self.cfg.camera.width
        self.camera_params.height = self.cfg.camera.height
        self.camera_params.horizontal_fov = self.cfg.camera.horizontal_fov
        self.camera_params.enable_tensors = True
        self.cameras = []
        for env_handle in self.envs:
            camera_handle = self.gym.create_camera_sensor(env_handle, self.camera_params)
            torso_handle = self.gym.get_actor_rigid_body_handle(env_handle, 0, self.torso_index)
            camera_offset = gymapi.Vec3(self.cfg.camera.offset[0], self.cfg.camera.offset[1], self.cfg.camera.offset[2])
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(self.cfg.camera.angle_randomization * (2 * np.random.random() - 1) + self.cfg.camera.angle))
            self.gym.attach_camera_to_body(camera_handle, env_handle, torso_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
            self.cameras.append(camera_handle)
            
    def post_process_camera_tensor(self):
        """
        First, post process the raw image and then stack along the time axis
        """
        new_images = torch.stack(self.cam_tensors)
        new_images = torch.nan_to_num(new_images, neginf=0)
        new_images = torch.clamp(new_images, min=-self.cfg.camera.far, max=-self.cfg.camera.near)
        # new_images = new_images[:, 4:-4, :-2] # crop the image
        self.last_visual_obs_buf = torch.clone(self.visual_obs_buf)
        self.visual_obs_buf = new_images.view(self.num_envs, -1)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

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
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
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
                # print(f"DOF {i} limits: {self.hard_dof_pos_limits[i, 0].item()}, {self.hard_dof_pos_limits[i, 1].item()}")
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
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
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
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
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] *= torch.abs(self.commands[env_ids, 2]) > 0.2


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller

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
    
    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, self.torso_link_index, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        
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
        noise_vec[(9 + 2 * self.num_dof):(9 + 2 * self.num_dof + self.num_actions)] = 0. # previous actions
        noise_vec[(9 + 2 * self.num_dof + self.num_actions + 15):(12 + 2 * self.num_dof + self.num_actions + 15)] = 0. # base lin vel
        
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        
        self.left_feet_pos = self.rigid_body_states[:, self.left_feet_indices, 0:3]
        self.right_feet_pos = self.rigid_body_states[:, self.right_feet_indices, 0:3]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.num_bodies, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.skill = self.cfg.asset.skill
        self.skill_init_prob = torch.tensor(self.cfg.asset.skill_init_prob, device=self.device)
        self.dir_buffers = torch.tensor([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]], dtype=torch.float, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.z_axis_unit = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0)
        self.ori_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.computed_torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.first_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,7:10])
        self.base_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,10:13])
        self.projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.gravity_vec)
        self.delay_buffer = torch.zeros(self.cfg.domain_rand.max_delay_timesteps, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        
        self.end_effector_pos = torch.concatenate((self.rigid_body_states[:, self.hand_pos_indices[0], :3],
                                                  self.rigid_body_states[:, self.hand_pos_indices[1], :3],
                                                  self.feet_pos[:, 0], self.feet_pos[:, 1],
                                                  self.rigid_body_states[:, self.head_index, :3]), dim=-1)
        self.end_effector_pos = self.end_effector_pos - self.root_states[:, :3].repeat(1, 5)
        for i in range(5):
            self.end_effector_pos[:, 3*i: 3*i+3] = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index, 3:7], self.end_effector_pos[:, 3*i: 3*i+3])

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_poses = self.default_dof_pos.repeat(self.num_envs,1)

        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.zero_force = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=self.device, requires_grad=False)

        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            self.com_displacement[:, 0] = self.com_displacement[:, 0] * 1.5
        if self.cfg.domain_rand.delay:
            self.delay_idx = torch.randint(low=0, high=self.cfg.domain_rand.max_delay_timesteps, size=(self.num_envs,), device=self.device)

        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        #joint powers
        self.joint_powers = torch.zeros(self.num_envs, 100, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        # create mocap dataset
        self.init_base_pos_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.init_base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.init_base_pos_xy[:] = self.base_init_state[:2] + self.env_origins[:, 0:2]
        self.init_base_quat[:] = self.base_init_state[3:7]

        motion_file = self.cfg.dataset.motion_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        joint_mapping_file = self.cfg.dataset.joint_mapping_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

        self.motionlib = MotionLib(motion_file=motion_file, 
                                   mapping_file=joint_mapping_file, 
                                   dof_names=self.dof_names,
                                   fps=self.cfg.dataset.frame_rate,
                                   device=self.device,
                                   window_length=self.cfg.amp.window_length, 
                                   ratio_random_range=self.cfg.amp.ratio_random_range)
        
        amp_obs_joint_id = []
        for i, name in enumerate(self.dof_names):
            if name in self.motionlib.mapping.keys():
                amp_obs_joint_id.append(i)
        self.amp_obs_joint_id = torch.tensor(amp_obs_joint_id, device=self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
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
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
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
        
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dof = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        left_foot_names = [s for s in body_names if self.cfg.asset.left_foot_name in s]
        right_foot_names = [s for s in body_names if self.cfg.asset.right_foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
            
        hand_pos_names = [s for s in body_names if self.cfg.asset.hand_pos_name in s]
        hand_colli_names = [s for s in body_names if self.cfg.asset.hand_colli_name in s]

        self.torso_link_index = body_names.index("torso_link")

        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            self.com_displacement[:, 0] = self.com_displacement[:, 0] * 1.5
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_gravity = torch.tensor(self.cfg.sim.gravity, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)

            # create robot asset & actor [0]
            pos = self.env_origins[i].clone()
            pos += torch.tensor([2.5, 0.0, 0.0], device=self.device)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                self.default_com_torso = copy.deepcopy(body_props[self.torso_link_index].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.actor_handles.append(actor_handle)

        self.left_hip_joint_indices = torch.zeros(len(self.cfg.control.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.control.left_hip_joints[i])
            
        self.right_hip_joint_indices = torch.zeros(len(self.cfg.control.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.control.right_hip_joints[i])
            
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))
            
        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.hand_pos_indices = torch.zeros(len(hand_pos_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hand_pos_names)):
            self.hand_pos_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hand_pos_names[i])
        
        self.hand_colli_indices = torch.zeros(len(hand_colli_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hand_colli_names)):
            self.hand_colli_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hand_colli_names[i])

        self.head_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.head_name)

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        left_feet_names = [s for s in body_names if self.cfg.asset.left_foot_name in s]
        right_feet_names = [s for s in body_names if self.cfg.asset.right_foot_name in s]
        
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        
        self.left_feet_indices = torch.zeros(len(left_feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_feet_names)):
            self.left_feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_feet_names[i])

        self.right_feet_indices = torch.zeros(len(right_feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_feet_names)):
            self.right_feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
            
        self.left_leg_joint_indices = torch.zeros(len(self.cfg.control.left_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_leg_joints)):
            self.left_leg_joint_indices[i] = self.dof_names.index(self.cfg.control.left_leg_joints[i])
            
        self.right_leg_joint_indices = torch.zeros(len(self.cfg.control.right_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_leg_joints)):
            self.right_leg_joint_indices[i] = self.dof_names.index(self.cfg.control.right_leg_joints[i])
            
        self.leg_joint_indices = torch.cat((self.left_leg_joint_indices, self.right_leg_joint_indices))
            
        self.left_arm_joint_indices = torch.zeros(len(self.cfg.control.left_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_arm_joints)):
            self.left_arm_joint_indices[i] = self.dof_names.index(self.cfg.control.left_arm_joints[i])
            
        self.right_arm_joint_indices = torch.zeros(len(self.cfg.control.right_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_arm_joints)):
            self.right_arm_joint_indices[i] = self.dof_names.index(self.cfg.control.right_arm_joints[i])
            
        self.arm_joint_indices = torch.cat((self.left_arm_joint_indices, self.right_arm_joint_indices))
            
        self.waist_joint_indices = torch.zeros(len(self.cfg.asset.waist_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.waist_joints)):
            self.waist_joint_indices[i] = self.dof_names.index(self.cfg.asset.waist_joints[i])
            
        self.ankle_joint_indices = torch.zeros(len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])

        self.upper_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.control.upper_body_link)

        self.keyframe_names = [s for s in body_names if self.cfg.asset.keyframe_name in s]
        self.keyframe_indices = torch.zeros(len(self.keyframe_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.keyframe_names):
            self.keyframe_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)


    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
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
    
    def _draw_debug_vis(self, x, y, z, clear=True):
        if clear:
            self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.2, 4, 4, None, color=(1, 0, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose) 


    #------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_vel(self):
        lin_vel_error = torch.sum(torch.square(self.cfg.rewards.target_vel * self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_yaw(self):
        rew = torch.exp(-torch.abs(self.commands[:,2] - self.yaw[:,0]))
        return rew

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1) / torch.clip(torch.sum(torch.square(self.commands[:,0:1]), dim=-1), min=0.01)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        return torch.abs(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_base_height_wrt_feet(self):
        # Penalize base height away from target
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
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques / self.p_gains.unsqueeze(0)), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_success_termination(self):
        # Terminal reward / penalty
        return self.success_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.computed_torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * self.first_contacts, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, 0:1], dim=1) > 0.1 # no reward for zero command
        return rew_airTime
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 3 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, 0:2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        rew_no_fly = 1.0 * single_contact
        rew_no_fly = torch.max(rew_no_fly, 1. * (torch.norm(self.commands[:, 0:1], dim=1) < 0.1)) # full reward for zero command
        return rew_no_fly
    
    def _reward_joint_tracking_error(self):
        return torch.sum(torch.square(self.joint_pos_target - self.dof_pos), dim=-1)
    
    def _reward_joint_deviation(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)
    
    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
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
        # return torch.clip(foot_leteral_dis - self.cfg.rewards.least_feet_distance_lateral, max=0)
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
        knee_lateral_dis = torch.abs(knee_pos_in_body_frame[:, 0, 1] - knee_pos_in_body_frame[:, 1, 1])
        return torch.clamp(knee_lateral_dis - self.cfg.rewards.least_knee_distance_lateral, max=0)
    
    def _reward_feet_distance_lateral(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        foot_leteral_dis = torch.abs(footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1])
        return torch.clamp(foot_leteral_dis - self.cfg.rewards.least_feet_distance_lateral, max=0)
    
    def _reward_feet_slip(self): 
        # Penalize feet slipping
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.sum(torch.norm(self.feet_vel[:,:,:2], dim=2) * contact, dim=1)

    def _reward_contact_momentum(self):
        # encourage soft contacts
        feet_contact_momentum_z = torch.abs(self.feet_vel[:, :, 2] * self.contact_forces[:, self.feet_indices, 2])
        return torch.sum(feet_contact_momentum_z, dim=1)
    
    def _reward_deviation_all_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)
    
    def _reward_deviation_arm_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.arm_joint_indices], dim=-1)
    
    def _reward_deviation_leg_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.leg_joint_indices], dim=-1)
    
    def _reward_deviation_hip_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.hip_joint_indices], dim=-1)
    
    def _reward_deviation_waist_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.waist_joint_indices], dim=-1)
    
    def _reward_deviation_ankle_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.ankle_joint_indices], dim=-1)

    def _reward_heading(self):
        return torch.exp(-0.5 * self.ori_error)