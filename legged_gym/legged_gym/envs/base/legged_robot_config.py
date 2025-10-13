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

from .base_config import BaseConfig
import numpy as np

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        
        num_actions = 29 # number of actuators on robot
        num_dofs = 29
        num_proprio_obs = 6 + num_dofs * 2 + num_actions + 3 * 5
        num_task_obs = 18
        num_actor_history = 6

        num_actor_obs = num_actor_history * (num_proprio_obs + num_task_obs)
        num_privileged_obs = num_proprio_obs + 3 + num_task_obs
        
        action_curriculum = False
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 5 # episode length in seconds

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "fast"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 0.9
        dynamic_friction = 0.9
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 10 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 40 # number of terrain rows (levels)
        num_cols = 40 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: sample heading command, iFalsef false: sample ang_vel_yaw
        heading_to_ang_vel = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.0, 0.8] # min max [m/s]
            lin_vel_y = [-0.0, 0.0] # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5] # min max [rad/s]
            heading = [-3.14, 3.14] # min max [rad]

    class init_state:
        pos = [0.0, 0.0, 0.75] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 150,
                     'knee': 300,
                     'ankle': 40,
                     'shoulder': 150,
                     'elbow': 150,
                     'waist_yaw': 150,
                     'wrist': 20,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'shoulder': 2,
                     'elbow': 2,
                     'waist_yaw': 2,
                     'wrist': 0.5,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        curriculum_joints = ['waist_yaw_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_shoulder_pitch_joint', 'left_elbow_joint', 'left_wrist_roll_joint', \
            'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_shoulder_pitch_joint', 'right_elbow_joint', 'right_wrist_roll_joint']
        left_leg_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint']
        right_leg_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']

        left_arm_joints = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint']
        right_arm_joints = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint']
        upper_body_link = "pelvis"  # "torso_link"

        left_hip_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint']
        right_hip_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint']

        hip_reduction = 1.0

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1_23dof_rev_1_0.urdf"
        name = "g1"
        hand_pos_name = "palm"
        hand_colli_name = "rubber_hand"
        foot_name = "ankle_pitch"
        head_name = "keyframe_head_link"
        left_foot_name = "left_foot"
        right_foot_name = "right_foot"
        penalize_contacts_on = ["hip", "knee", "torso", "shoulder", "elbow", "pelvis", "hand"]
        terminate_after_contacts_on = []

        waist_joints = ["waist_yaw_joint"]
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        ankle_joints = [ "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_wrist_roll_joint", "right_wrist_roll_joint"]
        upper_body_link = "torso_link"
        imu_link = "imu_link"
        knee_names = ["left_knee_link", "right_knee_link"]
        
        keyframe_name = "keyframe"

        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

        class box:
            pass
        class chair:
            pass
        class camera:
            pass

    class domain_rand:
        use_random = False
        
        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]
        
        randomize_motor_strength = use_random
        motor_strength_range = [0.9, 1.1]

        randomize_payload_mass = use_random
        payload_mass_range = [-5, 10]

        randomize_com_displacement = use_random
        com_displacement_range = [-0.1, 0.1]

        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = use_random
        friction_range = [0.1, 1.5]
        
        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]
        
        randomize_kp = use_random
        kp_range = [0.85, 1.15]
        
        randomize_kd = use_random
        kd_range = [0.85, 1.15]
        
        randomize_initial_joint_pos = use_random
        initial_joint_pos_scale = [1.0, 1.0]
        initial_joint_pos_offset = [-0.1, 0.1]
        
        push_robots = use_random
        push_interval_s = 8
        max_push_vel_xy = 0.5

        disturbance = use_random
        disturbance_interval = 5
        disturbance_range = [-100, 100]

        delay = use_random
        max_delay_timesteps = 5

    class rewards:
        class scales:
            action_rate = -0.
            action_vanish = -0.
            ang_vel_xy = -0.
            base_height = -0.
            base_height_wrt_feet = 0.
            collision = -0.
            contact_momentum = 0.
            deviation_all_joint = -0.
            deviation_arm_joint = -0.
            deviation_leg_joint = -0.
            deviation_hip_joint = -0.
            deviation_waist_joint = -0.
            deviation_ankle_joint = -0.
            dof_acc = -0.
            dof_pos_limits = -0.
            dof_vel = -0.
            dof_vel_limits = -0.
            feet_air_time = 0.
            feet_clearance = -0.
            feet_contact_forces = -0.
            feet_distance_lateral = 0.
            feet_ground_parallel = -0.
            feet_parallel = -0.
            feet_slip = -0.
            feet_stumble = -0.
            foothold = -0.
            foothold_first_contact = 0.
            joint_power = -0.
            joint_tracking_error = -0.
            knee_distance_lateral = 0.
            lin_vel_z = -0.
            no_fly = 0.
            orientation = -0.
            smoothness = -0.
            stand_still = -0.
            termination = -0.
            termination_foothold = -0.
            torques = -0.
            tracking_ang_vel = 0.
            tracking_lin_vel = 0.

        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.95
        base_height_target = 0.75
        head_height_target = 1.15
        max_contact_force = 350. # forces above this value are penalized
        clearance_height_target = -0.66
        least_feet_distance = 0.18
        least_feet_distance_lateral = 0.18
        least_knee_distance_lateral = 0.18

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'HIMOnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 256]
        critic_hidden_dims = [512, 256, 256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100 # per iteration
        max_iterations = 200000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt