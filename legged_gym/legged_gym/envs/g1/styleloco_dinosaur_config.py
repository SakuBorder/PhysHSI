from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        
        num_actions = 29 # number of actuators on robot
        num_dofs = 29
        num_one_step_proprio_obs = 3 + 6 + num_dofs * 2 + num_actions + 3 * 5
        num_actor_history = 6
        num_actor_obs = num_actor_history * num_one_step_proprio_obs
        num_privileged_obs = num_one_step_proprio_obs + 3

        env_spacing = 10.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 15 # episode length in seconds

        action_curriculum = False
        
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 8. # time before command are changed[s]
        heading_command = False # if true: sample heading command, iFalsef false: sample ang_vel_yaw
        heading_to_ang_vel = True # if true: compute ang vel command from heading error
        
        lin_vel_clip = 0.2
        ang_vel_clip = 0.2

        class ranges:
            lin_vel_x = [-0.8, 0.8] # min max [m/s]
            lin_vel_y = [-0.8, 0.8] # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5] # min max [rad/s]
            heading = [-3.14, 3.14] # min max [rad]

    class init_state(LeggedRobotCfg.init_state):
        pos = [2.3, 0.0, 0.8] # x,y,z [m]
        rot = [0.0, 0.0, 1.0, 0.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            
            'left_hip_pitch_joint': -0.1, # in use
            'left_hip_roll_joint': 0.0, # in use
            'left_hip_yaw_joint': 0.0, # in use
            'left_knee_joint': 0.3, # in use
            'left_ankle_pitch_joint': -0.2, # in use
            'left_ankle_roll_joint': 0.0,

            'right_hip_pitch_joint': -0.1, # in use
            'right_hip_roll_joint': 0.0, # in use
            'right_hip_yaw_joint': 0.0, # in use
            'right_knee_joint': 0.3, # in use
            'right_ankle_pitch_joint': -0.2, # in use
            'right_ankle_roll_joint': 0.0,

            'waist_yaw_joint': 0.0, # in use
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,

            'left_shoulder_pitch_joint': 0.0, # in use
            'left_shoulder_roll_joint': 0.1, # in use
            'left_shoulder_yaw_joint': 0.0, # in use
            'left_elbow_joint': 1.2, # in use
            'left_wrist_roll_joint': 0.0, # in use
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,

            'right_shoulder_pitch_joint': 0.0, # in use
            'right_shoulder_roll_joint': -0.1,  # in use
            'right_shoulder_yaw_joint': 0.0, # in use
            'right_elbow_joint': 1.2,  # in use
            'right_wrist_roll_joint': 0.0, # in use
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
            }
 

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 150,
                     'knee': 300,
                     'ankle': 40,
                     
                     "waist_yaw": 300,
                     "waist_roll": 300,
                     "waist_pitch": 300,

                     "shoulder": 200,
                     "elbow": 100,
                     "wrist": 20,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 1,

                     "waist_yaw": 4,
                     "waist_roll": 4,
                     "waist_pitch": 4,

                     "shoulder": 3,
                     "elbow": 1,
                     "wrist": 0.5,
                     }  # [N*m/rad]
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


    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "fast"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 0.8
        dynamic_friction = 0.7
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
        ankle_joints = [ "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_wrist_roll_joint", "right_wrist_roll_joint"]
        upper_body_link = "torso_link"
        imu_link = "imu_link"
        knee_names = ["left_knee_link", "right_knee_link"]
        
        keyframe_name = "keyframe"

        disable_gravity = False
        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False

        density = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.01

        reset_mode = 'hybrid' # 'default', 'random', 'hybrid'
        hybrid_init_prob = 0.5

        skill = ['loco']
        skill_init_prob = [1.0]

    class domain_rand(LeggedRobotCfg.domain_rand):
        use_random = True
        
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
        max_push_vel_xy = 0.2

        disturbance = use_random
        disturbance_interval = 5
        disturbance_range = [-50, 50]

        delay = use_random
        max_delay_timesteps = 6
        
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # regularization rewards
            dof_acc = -1e-7
            action_rate = -0.03
            torques = -1e-4
            dof_vel = -1e-4
            dof_pos_limits = -5.0
            dof_vel_limits = -1e-3
            torque_limits = -0.1

            # task rewards
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5

    class normalization:
        class obs_scales:
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            lin_vel = 2.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            ang_vel = 0.3
            gravity = 0.05
            dof_pos = 0.02
            dof_vel = 2.0
            end_effector = 0.05
            lin_vel = 0.1

    class dataset:
        motion_file = "{LEGGED_GYM_ROOT_DIR}/resources/config/styleloco_dinosaur.yaml"
        joint_mapping_file = "{LEGGED_GYM_ROOT_DIR}/resources/config/joint_id.txt"
        frame_rate = 60
        min_time = 0.1 # [s]

    class amp:
        amp_coef = 0.3
        num_one_step_obs = 1 + 29 + 5 * 3 + 3 + 3 + 6
        window_length = 10
        num_obs = num_one_step_obs * window_length
        ratio_random_range = [0.9, 1.1]
        use_normalizer = False

    class viewer:
        ref_env = 0
        pos = [10, -5, 4]  # [m]
        lookat = [11., 3, 2.]  # [m]

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

class G1CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100 # per iteration
        max_iterations = 30000 # number of policy updates
        use_muon_optim = False

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        run_name = 'styleloco_dinosaur_coef0.3'
        experiment_name = 'amp_styleloco'
        logger = 'tensorboard'  # ['tensorboard', 'wandb']
        wandb_project = 'amp_styleloco'
        wandb_entity = 'YOUR_ENTITY_NAME'  # set to your wandb entity name here
        
        # load and resume
        resume = False
        resume_path = None

    amp = G1Cfg.amp