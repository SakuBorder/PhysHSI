from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1MultiTaskCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096

        num_actions = 29
        num_dofs = 29

        enable_task_mask_obs = True

        # 修正：本体观测 = ang_vel(3) + gravity(3) + dof_pos(29) + dof_vel(29) + ee_pos(15) + actions(29) = 108
        num_one_step_proprio_obs = 6 + num_dofs * 2 + num_actions + 3 * 5

        task_obs_dim_map = {
            "traj": 2 * 10,
            "sitdown": 6,
            "carrybox": 9,
            "standup": 5,
        }
        num_task_obs = sum(task_obs_dim_map.values()) + len(task_obs_dim_map)

        num_actor_history = 6
        num_actor_obs = num_actor_history * (num_one_step_proprio_obs + num_task_obs)

        # 修正：critic obs = 本体(108) + lin_vel(3) + 任务
        num_privileged_obs = num_one_step_proprio_obs + 3 + num_task_obs

        env_spacing = 10.
        send_timeouts = True
        episode_length_s = 15

        action_curriculum = False

        task_init_prob = [0.25, 0.25, 0.25, 0.25]
        task_names = ["traj", "sitdown", "carrybox", "standup"]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 8.
        heading_command = False
        heading_to_ang_vel = True

        lin_vel_clip = 0.0
        ang_vel_clip = 0.0

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [-3.14, 3.14]

    class traj:
        num_samples = 10
        sample_timestep = 0.5
        num_vertices = 101
        dtheta_max = 2.0
        speed_min = 0.0
        speed_max = 0.8
        accel_max = 1.0
        sharp_turn_prob = 0.15
        sharp_turn_angle = 1.57
        fail_dist = 4.0
        
        # ✅ 修改：只使用 YAML 中存在的技能
        skill = ["loco"]
        skill_init_prob = [1.0]

    class multi_task:
        task_names = ["traj", "sitdown", "carrybox", "standup"]
        task_init_prob = [0.25, 0.25, 0.25, 0.25]
        task_obs_dim = {
            "traj": 20,
            "sitdown": 6,
            "carrybox": 9,
            "standup": 5,
        }

        class sitdown:
            distance_range = [1.2, 3.0]
            lateral_noise = 1.0
            height_range = [0.35, 0.55]
            facing_noise = 0.3
            
            # ✅ 修改：使用正确的技能名称
            skill = ["sitDown", "loco"]
            skill_init_prob = [0.3, 0.7]

        class carrybox:
            start_distance_range = [0.6, 1.8]
            goal_distance_range = [1.0, 2.5]
            height = 0.5
            
            # ✅ 修改：只使用 YAML 中存在的技能
            skill = ["pickUp", "carryWith", "putDown", "loco"]
            skill_init_prob = [0.25, 0.25, 0.25, 0.25]

        class standup:
            target_height = 0.95
            up_tolerance = 0.2
            
            # ✅ 修改：使用正确的技能名称（standup 小写）
            skill = ["standup", "loco"]
            skill_init_prob = [0.3, 0.7]

    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            dof_acc = -1e-7
            action_rate = -0.03
            torques = -1e-4
            dof_vel = -2e-4
            dof_pos_limits = -5.0
            dof_vel_limits = -1e-3
            torque_limits = -0.03

            loco_task = 1.0
            sitDown_task = 1.0
            walk_task = 1.0
            carryup_task = 1.0
            relocation_task = 1.0
            standup_task = 3.0

        robot2chair_vel = 1.0
        loco_heading = 1.0
        sit_pos_far = 1.0
        sit_pos_near = 1.0
        sit_height = 1.0
        sit_heading = 1.0
        target_speed_loco = 0.85
        thresh_robot2chair = 0.7

        robot2object_pos = 0.0
        robot2object_vel = 1.0
        start_heading = 0.5
        hand_pos = 0.7
        box_height = 2.0
        relocation_heading = 0.5
        relocation_heading_vel = 0.0
        robot2goal_pos = 0.0
        robot2goal_vel = 1.0
        object2goal_pos = 1.0
        put_box = 1.0
        target_speed_carry = 0.85
        target_box_height = 0.72
        thresh_robot2object = 0.7
        thresh_robot2goal = 0.65
        thresh_object2goal = 0.05
        thresh_object2start = 0.5

        base_height = 0.0
        head_height = 0.5
        stand_still = 1.0
        hand_free = 0.5
        pos_near = 0.2

    class init_state(LeggedRobotCfg.init_state):
        pos = [2.3, 0.0, 0.8]
        rot = [0.0, 0.0, 1.0, 0.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            'left_hip_pitch_joint': -0.1,
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,

            'right_hip_pitch_joint': -0.1,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,

            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,

            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.1,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.2,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,

            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': -0.1,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 1.2,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            'hip_yaw': 150,
            'hip_roll': 150,
            'hip_pitch': 150,
            'knee': 300,
            'ankle': 40,
            'waist_yaw': 300,
            'waist_roll': 300,
            'waist_pitch': 300,
            'shoulder': 200,
            'elbow': 100,
            'wrist': 20,
        }
        damping = {
            'hip_yaw': 2,
            'hip_roll': 2,
            'hip_pitch': 2,
            'knee': 4,
            'ankle': 1,
            'waist_yaw': 4,
            'waist_roll': 4,
            'waist_pitch': 4,
            'shoulder': 3,
            'elbow': 1,
            'wrist': 0.5,
        }
        action_scale = 0.25
        decimation = 4
        curriculum_joints = ['waist_yaw_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 
                             'left_shoulder_pitch_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 
                             'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                             'right_shoulder_pitch_joint', 'right_elbow_joint', 'right_wrist_roll_joint']
        left_leg_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint',
                           'left_ankle_pitch_joint', 'left_ankle_roll_joint']
        right_leg_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint',
                            'right_ankle_pitch_joint', 'right_ankle_roll_joint']
        left_arm_joints = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                           'left_elbow_joint', 'left_wrist_roll_joint']
        right_arm_joints = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                            'right_elbow_joint', 'right_wrist_roll_joint']
        upper_body_link = "torso_link"
        left_hip_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint']
        right_hip_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint']

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1_29dof.urdf"
        name = "g1"
        hand_pos_name = "palm_link"
        hand_colli_name = "rubber_hand"
        foot_name = "ankle_pitch_link"
        head_name = "mid360_link"
        pelvis_contact_name = "pelvis_contact_link"
        left_foot_name = "left_foot"
        right_foot_name = "right_foot"
        penalize_contacts_on = ["hip", "knee", "torso", "shoulder", "pelvis"]
        terminate_after_contacts_on = []
        waist_joints = ["waist_yaw_joint"]
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        ankle_joints = ["left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_wrist_roll_joint", "right_wrist_roll_joint"]
        upper_body_link = "torso_link"
        imu_link = "imu_link"
        knee_names = ["left_knee_link", "right_knee_link"]
        keyframe_name = "keyframe"
        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.01
        
        reset_mode = 'hybrid'
        hybrid_init_prob = 0.5
        
        # ✅ 修改：只使用 YAML 中存在的技能
        skill = ['loco', 'sitDown', 'pickUp', 'carryWith', 'putDown', 'standup']
        skill_init_prob = [0.3, 0.15, 0.15, 0.15, 0.1, 0.15]

    class marker:
        class asset:
            file = "{LEGGED_GYM_ROOT_DIR}/resources/objects/location_marker.urdf"
            name = "traj_marker"
        disable_gravity = True
        height_offset = 0.05

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            ang_vel = 0.3
            gravity = 0.05
            dof_pos = 0.02
            dof_vel = 2.0
            end_effector = 0.05
            lin_vel = 0.1

    class normalization:
        class obs_scales:
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            lin_vel = 2.0
        clip_observations = 100.
        clip_actions = 100.

    class dataset:
        motion_file = "{LEGGED_GYM_ROOT_DIR}/resources/config/multi_task.yaml"
        joint_mapping_file = "{LEGGED_GYM_ROOT_DIR}/resources/config/joint_id.txt"
        frame_rate = 60
        min_time = 0.1

    class amp:
        amp_coef = 0.3
        num_one_step_obs = 1 + 29 + 5 * 3 + 3 + 3 + 6
        window_length = 10
        num_obs = num_one_step_obs * window_length
        ratio_random_range = [0.9, 1.1]
        use_normalizer = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_random = False
        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]
        randomize_motor_strength = use_random
        motor_strength_range = [0.9, 1.1]
        randomize_payload_mass = use_random
        payload_mass_range = [-5, 10]
        randomize_com_displacement = use_random
        com_displacement_range = [-0.02, 0.02]
        randomize_link_mass = use_random
        link_mass_range = [0.9, 1.1]
        randomize_friction = use_random
        friction_range = [0.1, 1.5]
        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]
        randomize_kp = use_random
        kp_range = [0.9, 1.1]
        randomize_kd = use_random
        kd_range = [0.9, 1.1]
        randomize_initial_joint_pos = use_random
        initial_joint_pos_scale = [1.0, 1.0]
        initial_joint_pos_offset = [-0.1, 0.1]
        disturbance = use_random
        disturbance_interval = 8
        disturbance_range = [-50, 50]
        delay = use_random
        max_delay_timesteps = 5
        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 0.1

    class viewer:
        ref_env = 0
        pos = [10, -5, 4]
        lookat = [11., 3, 2.]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 8
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24
            default_buffer_size_multiplier = 5
            contact_collection = 2


class G1MultiTaskCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        
        transformer_params = {
            "num_features": 128,
            "tokenizer_units": [256, 128],
            "drop_ratio": 0.1,
            "layer_num_heads": 4,
            "layer_dim_feedforward": 256,
            "num_layers": 2,
            "extra_mlp_units": [128, 64],
            "use_pos_embed": True,
        }

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'TransformerActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100
        max_iterations = 20000
        use_muon_optim = False
        
        save_interval = 500
        run_name = 'multitask'
        experiment_name = 'amp_multitask'
        logger = 'tensorboard'
        wandb_project = 'amp_multitask'
        wandb_entity = 'YOUR_ENTITY_NAME'
        
        resume = False
        resume_path = None
    
    # ✅ 关键：引用 G1MultiTaskCfg 中的 amp 配置
    amp = G1MultiTaskCfg.amp