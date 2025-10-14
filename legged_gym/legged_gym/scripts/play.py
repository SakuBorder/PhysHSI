import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, export_jit_to_onnx, load_onnx_policy, task_registry, Logger

import numpy as np
import torch
import keyboard
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.env.test = True
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.delay = False
    env_cfg.domain_rand.push_robots = False
    train_cfg.runner.resume = True

    # carrybox
    if args.task == 'carrybox':
        env_cfg.asset.box.random_props = False
        env_cfg.asset.box.reset_mode = 'default'
        env_cfg.env.episode_length_s = 10
    # sitdown
    if args.task == 'sitdown' or args.task == 'liedown':
        env_cfg.asset.chair.random_size = False
        env_cfg.asset.chair.reset_mode = 'default'
    # styleloco
    if args.task == 'styleloco_dinosaur' or args.task == 'styleloco_highknee':
        env_cfg.terrain.mesh_type = 'plane'
        env_cfg.terrain.num_rows = 3
        env_cfg.terrain.num_cols = 3
    
    if args.play_dataset:
        train_cfg.runner.resume = False
        env_cfg.viewer.pos = [-5, -5, 4]
        env_cfg.viewer.lookat = [0, 0, 2.]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit & onnx module (used to run it from C++)
    if EXPORT_POLICY:
        policy_name = 'policy_name'
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, policy_name)
        print('Exported policy as jit script to: ', path)

        jit_path = os.path.join(path, f'{policy_name}.pt')
        jit_model = torch.jit.load(jit_path)
        dummy_input = torch.randn(1, obs.shape[1], device='cpu')
        onnx_path = os.path.join(path, f'{policy_name}.onnx')
        export_jit_to_onnx(jit_model, onnx_path, dummy_input)
        policy = load_onnx_policy(onnx_path)

    for i in range(10*int(env.max_episode_length)):
        env.commands[:, 0] = 0.8
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        result = env.gym.fetch_results(env.sim, True)
        actions = policy(obs.detach())
        if args.play_dataset:
            env.play_dataset_step(i)
        else:
            obs, _, rews, dones, infos, _, _, amp_state = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
