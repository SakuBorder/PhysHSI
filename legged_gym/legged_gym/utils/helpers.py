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
from isaacgym import gymapi
from isaacgym import gymutil
import os
import copy

import numpy as np
import random

import torch.nn.functional as F
import onnxruntime as ort
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
import torch


def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

# def get_load_path(root, load_run=-1, checkpoint=-1):
#     try:
#         runs = os.listdir(root)
#         #TODO sort by date to handle change of month
#         runs.sort()
#         if 'exported' in runs: runs.remove('exported')
#         last_run = os.path.join(root, runs[-1])
#     except:
#         raise ValueError("No runs in this directory: " + root)
#     if load_run==-1:
#         load_run = last_run
#     else:
#         load_run = os.path.join(root, load_run)
#     print(load_run)
#     if checkpoint==-1:
#         models = [file for file in os.listdir(load_run) if 'model' in file]
#         models.sort(key=lambda m: '{0:0>15}'.format(m))
#         model = models[-1]
#     else:
#         model = "model_{}.pt".format(checkpoint) 

#     load_path = os.path.join(load_run, model)
#     return load_path

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed
        if args.play_dataset is not None:
            env_cfg.play_dataset = args.play_dataset
        else:
            env_cfg.play_dataset = False
        if env_cfg.play_dataset:
            env_cfg.env.num_envs = 1
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.resume_path is not None:
            cfg_train.runner.resume_path = args.resume_path
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "g1", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--resume_path", "type": str,  "help": "Path to the directory where the run to be resumed is located."},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        {"name": "--exptid", "type": str, "help": "exptid"},
        {"name": "--resumeid", "type": str, "help": "exptid"},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--play_dataset", "action": "store_true", "default": False, "help": "Visualize the reference."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    # args.sim_device_id = args.compute_device_id
    args.sim_device = args.rl_device
    # if args.sim_device=='cuda':
    #     args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path, policy_name):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f'{policy_name}.pt')

    model = copy.deepcopy(actor_critic.actor).to('cpu')
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)
        
def load_onnx_policy(path):
    model = ort.InferenceSession(path)
    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cuda:0")
    return run_inference

def export_jit_to_onnx(jit_model, path, dummy_input):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.onnx.export(
        jit_model,                  # JIT 模型
        dummy_input,                # 示例输入张量
        path,                       # 输出文件路径
        export_params=True,         # 导出训练好的参数
        opset_version=11,           # ONNX opset version（推荐 11 或更高版本）
        do_constant_folding=True,   # 是否启用常量折叠优化
        input_names=['input'],      # 输入节点名称
        output_names=['output'],    # 输出节点名称
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # 动态轴支持
    )
    print(f"Exported JIT model to ONNX at: {path}")

def export_policy_as_onnx(actor_critic, path):
    if hasattr(actor_critic, 'estimator'):
        os.makedirs(path, exist_ok=True)
        if not actor_critic.blind:
            model = Policy_with_HeightOnnx(actor_critic)
            model.export(path, 'policy.onnx')
            estmodel = Return_Onnx(actor_critic)
            estmodel.export(path, 'estreturn.onnx')
        else:
            model = PolicyOnnx(actor_critic)
            model.export(path, 'policy.onnx')
            estmodel = Return_Onnx(actor_critic)
            estmodel.export(path, 'estreturn.onnx')
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)



        
def export_jit(model, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'policy.pt')
    model.to('cpu')
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


class PolicyHIMJIT(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.history_length = actor_critic.actor_history_length
        self.num_one_step_obs = actor_critic.num_one_step_obs
        self.num_proprioceptive_obs = self.history_length * self.num_one_step_obs
        self.new_obs_start = self.num_one_step_obs * (self.history_length - 1)

    def forward(self, obs_history):
        if obs_history.size(1) != self.num_proprioceptive_obs:
            print('Expected size: ', self.num_proprioceptive_obs)
            print('Got size: ', obs_history.size(1))
            raise ValueError("Input tensor has wrong size")
        parts = self.estimator(obs_history)
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        return self.actor(torch.cat((obs_history[:, self.new_obs_start:], vel, z), dim=1))
    
    def export(self, path, filename):
        self.to("cpu")
        
    
class PolicyOnnx(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimatorenc = copy.deepcopy(actor_critic.estimator.encoder)
        self.estimatorvel = copy.deepcopy(actor_critic.estimator.vel_mu)
        self.history_length = actor_critic.actor_history_length
        self.num_one_step_obs = actor_critic.num_one_step_obs
        self.num_proprioceptive_obs = self.history_length * self.num_one_step_obs
        self.new_obs_start = self.num_one_step_obs * (self.history_length - 1)

    def forward(self, x):
        encoded = self.estimatorenc(x)
        vel = self.estimatorvel(encoded)
        return self.actor(torch.cat((x[:, self.new_obs_start:], vel), dim=1))

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.num_proprioceptive_obs)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=True,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )    

class PolicyHIM_with_Height(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.terrain_encoder = copy.deepcopy(actor_critic.terrain_encoder)
        self.history_length = actor_critic.actor_history_length
        self.num_one_step_obs = actor_critic.num_one_step_obs
        self.num_height_points = actor_critic.num_height_points
        self.num_proprioceptive_obs = self.history_length * self.num_one_step_obs
        self.new_obs_start = self.num_one_step_obs * (self.history_length - 1)
        self.input_obs_dim = self.num_proprioceptive_obs + self.num_height_points

    def forward(self, obs_history):
        if obs_history.size(1) != self.input_obs_dim:
            print('Expected size: ', self.input_obs_dim)
            print('Got size: ', obs_history.size(1))
            raise ValueError("Input tensor has wrong size")
        parts = self.estimator(obs_history[:, 0:self.num_proprioceptive_obs])
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        terrain_latent = self.terrain_encoder(obs_history[:, self.new_obs_start:])
        return self.actor(torch.cat((obs_history[:, self.new_obs_start:self.num_proprioceptive_obs], vel, z, terrain_latent), dim=1))
    
class Policy_with_HeightOnnx(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimatorenc = copy.deepcopy(actor_critic.estimator.encoder)
        self.estimatorvel = copy.deepcopy(actor_critic.estimator.vel_mu)
        self.terrain_encoder = copy.deepcopy(actor_critic.terrain_encoder)
        self.history_length = actor_critic.actor_history_length
        self.num_one_step_obs = actor_critic.num_one_step_obs
        self.num_actor_preception = actor_critic.num_actor_preception
        self.num_obs = self.history_length * self.num_one_step_obs
        self.new_obs_start = self.num_one_step_obs * (self.history_length - 1)
        self.input_obs_dim = self.num_obs
        
    def forward(self, x):
        encoded = self.estimatorenc(x)
        vel = self.estimatorvel(encoded)
        terrain_latent = self.terrain_encoder(x[:,-self.num_one_step_obs:])
        return self.actor(torch.cat((x[:, -self.num_one_step_obs:-self.num_actor_preception], vel, terrain_latent), dim=1))
    
    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.input_obs_dim)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=True,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )

class Return_Onnx(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.rtestimator = copy.deepcopy(actor_critic.estcritic)

        self.history_length = actor_critic.actor_history_length
        self.num_one_step_obs = actor_critic.num_one_step_obs
        self.num_actor_preception = actor_critic.num_actor_preception

        self.num_one_step_prop_obs = self.num_one_step_obs - self.num_actor_preception
        self.num_prop_obs = self.history_length * self.num_one_step_prop_obs
 
        self.new_obs_start = self.num_one_step_obs * (self.history_length - 1)
        self.input_obs_dim = self.num_prop_obs
        
    def forward(self, x):
        return self.rtestimator(x)
    
    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.input_obs_dim)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=True,
            input_names=["propobs"],
            output_names=["estvalues"],
            dynamic_axes={},
        )
        
    