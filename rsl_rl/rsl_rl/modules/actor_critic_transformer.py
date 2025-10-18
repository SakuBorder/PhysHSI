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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .transformer_modelling import DecisionTransformer


class TransformerActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        actor_history_length,
        num_actions=19,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        device="cpu",
        # Multi-task parameters
        self_obs_size=None,
        task_obs_size=None,
        multi_task_info=None,
        transformer_params=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(TransformerActorCritic, self).__init__()

        activation = get_activation(activation)

        self.actor_history_length = actor_history_length
        self.long_history_length = actor_history_length
        self.obs_dim = num_actor_obs
        self.action_dim = num_actions

        self.device = device

        # Multi-task support
        self.enable_multi_task = multi_task_info is not None and transformer_params is not None

        if self.enable_multi_task:
            inferred_task_obs = task_obs_size if task_obs_size is not None else 0
            inferred_self_obs = self_obs_size if self_obs_size is not None else max(
                num_actor_obs - inferred_task_obs, 0
            )
            self.self_obs_size = int(inferred_self_obs)
            self.task_obs_size = int(task_obs_size if task_obs_size is not None else inferred_task_obs)
            self.step_obs_dim = self.self_obs_size + self.task_obs_size

            self.multi_task_info = multi_task_info
            if not self.multi_task_info.get("enable_task_mask_obs", False):
                raise AssertionError("Multi-task transformer requires task mask observations")

            self.task_obs_onehot_size = int(self.multi_task_info["onehot_size"])
            self.task_obs_tota_size = int(self.multi_task_info["tota_subtask_obs_size"])

            raw_sizes = self.multi_task_info["each_subtask_obs_size"]
            if isinstance(raw_sizes, torch.Tensor):
                raw_sizes = raw_sizes.cpu().tolist()
            self.task_obs_each_size = [int(s) for s in raw_sizes]

            raw_indices = self.multi_task_info["each_subtask_obs_indx"]
            if isinstance(raw_indices, torch.Tensor):
                raw_indices = raw_indices.cpu().tolist()
            self.task_obs_each_indx = [int(idx) for idx in raw_indices]

            self.each_subtask_names = list(self.multi_task_info["each_subtask_name"])

            if not self.task_obs_each_indx or self.task_obs_each_indx[-1] != self.task_obs_tota_size:
                raise AssertionError("Task observation metadata inconsistent with total size")

            print(f"Multi-task enabled with {self.task_obs_onehot_size} tasks")
            print(f"Task names: {self.each_subtask_names}")
        else:
            self.self_obs_size = num_actor_obs
            self.task_obs_size = 0
            self.step_obs_dim = self.self_obs_size

        # ✅ 关键修改：Critic 使用单步观测维度，不是历史堆叠的
        mlp_input_dim_c = num_critic_obs
        
        # 调试信息
        print(f"🔍 Debug Info:")
        print(f"   - Actor obs dim (with history): {num_actor_obs}")
        print(f"   - Critic obs dim (single step): {num_critic_obs}")
        print(f"   - Critic input dimension: {mlp_input_dim_c}")
        if self.enable_multi_task:
            print(f"   - Self obs size: {self.self_obs_size}")
            print(f"   - Task obs size: {self.task_obs_size}")

        # Policy - choose between Decision Transformer or Multi-task Transformer
        if self.enable_multi_task and transformer_params is not None:
            print("Building Multi-task Transformer Actor")
            self._build_multitask_transformer_actor(transformer_params, num_actions, activation)
        else:
            print("Building Decision Transformer Actor")
            self.actor = DecisionTransformer(
                state_dim=num_actor_obs - num_actions,
                act_dim=num_actions,
                n_blocks=1,
                h_dim=64,
                context_len=self.long_history_length,
                n_heads=4,
                drop_p=0.1,
            )

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor Structure: {self.actor if not self.enable_multi_task else 'Multi-task Transformer'}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _build_multitask_transformer_actor(self, transformer_params, num_actions, activation):
        """Build multi-task transformer actor following file 2's architecture"""

        num_features = transformer_params.get("num_features", 64)
        num_tokens = 1 + len(self.task_obs_each_size) + 1  # weight + self + multiple tasks
        drop_ratio = transformer_params.get("drop_ratio", 0.0)
        tokenizer_units = transformer_params.get("tokenizer_units", [256, 128])

        self.token_feature_dim = num_features
        self.transformer_num_tokens = num_tokens
        
        print("Building tokenizer for self obs")
        self.self_encoder = self._build_mlp(
            input_size=self.self_obs_size,
            units=tokenizer_units + [num_features],
            activation=activation
        )
        
        self.task_encoder = nn.ModuleList()
        for idx, task_size in enumerate(self.task_obs_each_size):
            print(f"Building tokenizer for subtask obs with size {task_size}")
            self.task_encoder.append(
                self._build_mlp(
                    input_size=task_size,
                    units=tokenizer_units + [num_features],
                    activation=activation
                )
            )
        
        # Initialize tokenizer weights
        for nets in [self.self_encoder, self.task_encoder]:
            for m in nets.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
        
        # Weight token for attention
        self.weight_token = nn.Parameter(torch.zeros(1, 1, num_features))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, num_features))
        self.pos_drop = nn.Identity()  # nn.Dropout(p=drop_ratio)
        self.use_pos_embed = transformer_params.get("use_pos_embed", True)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=transformer_params.get("layer_num_heads", 4),
            dim_feedforward=transformer_params.get("layer_dim_feedforward", 256),
            dropout=drop_ratio,
            activation='relu',
            batch_first=True,
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=transformer_params.get("num_layers", 2)
        )
        
        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.weight_token, std=0.02)
        
        # Output composer/head
        extra_mlp_units = transformer_params.get("extra_mlp_units", [128, 64])
        self.composer = self._build_mlp(
            input_size=num_features,
            units=extra_mlp_units + [num_actions],
            activation=activation
        )
        
        # Mark as multi-task transformer
        self.actor = None  # We'll use the transformer components directly

    def _build_mlp(self, input_size, units, activation):
        """Helper function to build MLP"""
        layers = []
        layers.append(nn.Linear(input_size, units[0]))
        layers.append(activation)
        
        for i in range(len(units) - 1):
            layers.append(nn.Linear(units[i], units[i + 1]))
            if i < len(units) - 2:  # Don't add activation after last layer
                layers.append(activation)
        
        return nn.Sequential(*layers)

    def _prepare_latest_observation(self, obs):
        """Extract the latest observation from history if needed"""
        if obs.dim() == 3:
            obs = obs[:, -1, :]
        if obs.shape[-1] > self.step_obs_dim:
            obs = obs[..., -self.step_obs_dim:]
        return obs

    def _eval_multitask_transformer(self, obs):
        """Evaluate multi-task transformer by tokenizing self and task observations."""

        obs = self._prepare_latest_observation(obs)
        B = obs.shape[0]

        # Split observations into proprioceptive and task-specific components
        self_obs = obs[..., :self.self_obs_size]
        self_token = self.self_encoder(self_obs).unsqueeze(1)  # (B, 1, num_feats)

        task_obs = obs[..., self.self_obs_size:]
        task_obs_real = task_obs[..., :self.task_obs_tota_size] if self.task_obs_tota_size > 0 else task_obs.new_zeros((B, 0))

        if self.task_obs_onehot_size > 0:
            tokens = []
            for i in range(self.task_obs_onehot_size):
                start, end = self.task_obs_each_indx[i], self.task_obs_each_indx[i + 1]
                tokens.append(self.task_encoder[i](task_obs_real[:, start:end]))
            task_token = torch.stack(tokens, dim=1)
        else:
            task_token = task_obs_real.new_zeros((B, 0, self.token_feature_dim))

        # Expand weight token
        weight_token = self.weight_token.expand(B, -1, -1)

        # Concatenate all tokens
        x = torch.cat((weight_token, self_token, task_token), dim=1)  # [B, num_tokens, num_feats]

        # Add positional embedding
        if self.use_pos_embed:
            x = self.pos_drop(x + self.pos_embed)

        # Compute key padding mask (True for padding positions)
        padding_mask = torch.ones((B, x.shape[1]), dtype=torch.bool, device=x.device)
        padding_mask[:, :2] = False  # weight and self tokens are always active

        if self.task_obs_onehot_size > 0:
            task_mask = task_obs[..., self.task_obs_tota_size:]
            task_obs_onehot_idx = task_mask.argmax(dim=-1) + 2
            task_obs_onehot_idx_mask = nn.functional.one_hot(
                task_obs_onehot_idx,
                num_classes=self.task_obs_onehot_size + 2,
            ).to(dtype=torch.bool)
            padding_mask[task_obs_onehot_idx_mask] = False

        # Torch 2.3's scaled_dot_product_attention kernel on CUDA expects a float mask
        # with `-inf` entries for the padded positions. Using a boolean mask triggers an
        # "invalid configuration argument" launch error. To keep CPU execution fast we
        # only convert to the float mask on CUDA.
        if x.is_cuda:
            src_key_padding_mask = padding_mask.float()
            src_key_padding_mask.masked_fill_(padding_mask, float("-inf"))
        else:
            src_key_padding_mask = padding_mask

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Output from weight token
        output = self.composer(x[:, 0])

        return output

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, cur_timestep=None):
        if self.enable_multi_task and self.actor is None:
            # Use multi-task transformer
            mean = self._eval_multitask_transformer(observations)
        else:
            # Use decision transformer
            mean = self.actor(
                cur_timestep, observations[:, :, :36], observations[:, :, 36:48]
            )
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, cur_timestep=None, **kwargs):
        self.update_distribution(observations, cur_timestep)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, cur_timestep=None):
        if self.enable_multi_task and self.actor is None:
            # Use multi-task transformer
            actions_mean = self._eval_multitask_transformer(observations)
        else:
            # Use decision transformer
            actions_mean = self.actor(
                cur_timestep, observations[:, :, :36], observations[:, :, 36:48]
            )
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """
        Evaluate the value function.
        
        Args:
            critic_observations: Can be 2D (batch, features) or 3D (batch, history, features)
        
        Returns:
            value: (batch, 1) value estimates
        """
        # ✅ 如果是 3D 张量（带历史），提取最后一帧
        if len(critic_observations.shape) == 3:
            critic_observations = critic_observations[:, -1, :]
        
        value = self.critic(critic_observations)
        return value

    def eval_actor(self, obs, cur_timestep=None):
        """Evaluate actor - compatible with both modes"""
        if self.enable_multi_task and self.actor is None:
            mu = self._eval_multitask_transformer(obs)
        else:
            mu = self.actor(cur_timestep, obs[:, :, :36], obs[:, :, 36:48])

        sigma = self.std
        return mu, sigma

    def eval_critic(self, obs):
        """Evaluate critic"""
        if len(obs.shape) == 3:
            # If history is provided, use latest observation
            obs = obs[:, -1, :]
        return self.critic(obs)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None