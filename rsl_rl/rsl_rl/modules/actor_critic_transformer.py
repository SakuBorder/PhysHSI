# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .transformer_modelling import DecisionTransformer

# ---- KISS: 彻底禁用易出错的 SDPA 快速内核，强制使用 math 实现 ----
# 如需恢复，可设环境变量 FORCE_MATH_SDPA=0
_FORCE_MATH_SDPA = os.environ.get("FORCE_MATH_SDPA", "1") != "0"
if _FORCE_MATH_SDPA:
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        # 低版本 PyTorch 没有这些开关就直接忽略
        pass


def _assert_bool_mask(mask: torch.Tensor, B: int, S: int, device: torch.device):
    assert isinstance(mask, torch.Tensor), "src_key_padding_mask must be a torch.Tensor"
    assert mask.dtype == torch.bool, f"src_key_padding_mask must be bool, got {mask.dtype}"
    assert mask.shape == (B, S), f"src_key_padding_mask shape must be {(B, S)}, got {tuple(mask.shape)}"
    assert mask.device == device, f"mask/device mismatch: mask={mask.device}, x={device}"
    assert S > 0, "sequence length S must be > 0"
    # 每个 batch 至少要有一个未屏蔽 token
    assert (~mask).any(dim=1).all(), "some batch has all tokens masked (all True in src_key_padding_mask)"


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

        act_mod = get_activation(activation)
        assert isinstance(act_mod, nn.Module), "activation must map to an nn.Module"

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

            if (not self.task_obs_each_indx) or (self.task_obs_each_indx[-1] != self.task_obs_tota_size):
                raise AssertionError("Task observation metadata inconsistent with total size")
            if len(self.task_obs_each_size) != self.task_obs_onehot_size:
                raise AssertionError("each_subtask_obs_size length must equal onehot_size")

            print(f"Multi-task enabled with {self.task_obs_onehot_size} tasks")
            print(f"Task names: {self.each_subtask_names}")
        else:
            self.self_obs_size = num_actor_obs
            self.task_obs_size = 0
            self.step_obs_dim = self.self_obs_size

        mlp_input_dim_c = num_critic_obs

        print(f"🔍 Debug Info:")
        print(f"   - Actor obs dim (with history): {num_actor_obs}")
        print(f"   - Critic obs dim (single step): {num_critic_obs}")
        print(f"   - Critic input dimension: {mlp_input_dim_c}")
        if self.enable_multi_task:
            print(f"   - Self obs size: {self.self_obs_size}")
            print(f"   - Task obs size: {self.task_obs_size}")

        # Policy
        if self.enable_multi_task and transformer_params is not None:
            print("Building Multi-task Transformer Actor")
            self._build_multitask_transformer_actor(transformer_params, num_actions, act_mod)
        else:
            print("Building Decision Transformer Actor")
            # KISS：保留你的 DT 默认配置
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
        critic_layers.append(act_mod)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])
                )
                critic_layers.append(act_mod)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor Structure: {self.actor if not self.enable_multi_task else 'Multi-task Transformer'}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def _build_multitask_transformer_actor(self, transformer_params, num_actions, activation_module):
        num_features = int(transformer_params.get("num_features", 64))
        num_tokens = 1 + len(self.task_obs_each_size) + 1  # weight + self + per-task
        drop_ratio = float(transformer_params.get("drop_ratio", 0.0))
        tokenizer_units = list(transformer_params.get("tokenizer_units", [256, 128]))

        self.token_feature_dim = num_features
        self.transformer_num_tokens = num_tokens

        print("Building tokenizer for self obs")
        self.self_encoder = self._build_mlp(
            input_size=self.self_obs_size,
            units=tokenizer_units + [num_features],
            activation=activation_module
        )

        self.task_encoder = nn.ModuleList()
        for idx, task_size in enumerate(self.task_obs_each_size):
            print(f"Building tokenizer for subtask obs with size {task_size}")
            self.task_encoder.append(
                self._build_mlp(
                    input_size=task_size,
                    units=tokenizer_units + [num_features],
                    activation=activation_module
                )
            )

        for nets in [self.self_encoder, self.task_encoder]:
            for m in nets.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

        self.weight_token = nn.Parameter(torch.zeros(1, 1, num_features))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, num_features))
        self.pos_drop = nn.Identity()
        self.use_pos_embed = bool(transformer_params.get("use_pos_embed", True))

        nhead = int(transformer_params.get("layer_num_heads", 4))
        assert num_features % nhead == 0, f"d_model ({num_features}) must be divisible by nhead ({nhead})"

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features,
            nhead=nhead,
            dim_feedforward=int(transformer_params.get("layer_dim_feedforward", 256)),
            dropout=drop_ratio,
            activation='relu',
            batch_first=True,  # 统一 batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(transformer_params.get("num_layers", 2))
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.weight_token, std=0.02)

        extra_mlp_units = list(transformer_params.get("extra_mlp_units", [128, 64]))
        self.composer = self._build_mlp(
            input_size=num_features,
            units=extra_mlp_units + [num_actions],
            activation=activation_module
        )

        self.actor = None  # 标记使用 transformer 分支

    def _build_mlp(self, input_size, units, activation):
        assert len(units) >= 1, "units must have at least one layer size"
        layers = []
        layers.append(nn.Linear(input_size, units[0]))
        layers.append(activation)

        for i in range(len(units) - 1):
            layers.append(nn.Linear(units[i], units[i + 1]))
            if i < len(units) - 2:
                layers.append(activation)

        return nn.Sequential(*layers)

    def _prepare_latest_observation(self, obs):
        # 支持 [B, T, D] 或 [B, D]
        if obs.dim() == 3:
            obs = obs[:, -1, :]
        if obs.shape[-1] > self.step_obs_dim:
            obs = obs[..., -self.step_obs_dim:]
        return obs

    def _eval_multitask_transformer(self, obs):
        # ------- 组装 token -------
        obs = self._prepare_latest_observation(obs)
        assert obs.dim() == 2, f"expect [B, D], got {tuple(obs.shape)}"
        B = obs.shape[0]
        device = obs.device

        # self token
        self_obs = obs[..., :self.self_obs_size]
        self_token = self.self_encoder(self_obs).unsqueeze(1)  # [B, 1, F]

        # task tokens
        task_obs = obs[..., self.self_obs_size:]  # [B, task_obs_size]
        task_obs_real = task_obs[..., :self.task_obs_tota_size] if self.task_obs_tota_size > 0 else task_obs.new_zeros((B, 0))

        if self.task_obs_onehot_size > 0:
            tokens = []
            for i in range(self.task_obs_onehot_size):
                start, end = self.task_obs_each_indx[i], self.task_obs_each_indx[i + 1]
                # 安全边界检查
                assert 0 <= start <= end <= task_obs_real.shape[-1], \
                    f"subtask slice out of range: [{start}, {end}) vs {task_obs_real.shape[-1]}"
                tokens.append(self.task_encoder[i](task_obs_real[:, start:end]))
            task_token = torch.stack(tokens, dim=1)  # [B, T, F], T=onehot_size
        else:
            task_token = task_obs_real.new_zeros((B, 0, self.token_feature_dim))

        # 拼 token：weight + self + tasks
        weight_token = self.weight_token.expand(B, -1, -1)  # [B, 1, F]
        x = torch.cat((weight_token, self_token, task_token), dim=1)  # [B, S, F]
        B2, S, F = x.shape
        assert B2 == B and F == self.token_feature_dim, "token stack shape mismatch"

        if self.use_pos_embed:
            assert self.pos_embed.shape[1] == S, f"pos_embed length {self.pos_embed.shape[1]} != S {S}"
            x = self.pos_drop(x + self.pos_embed)  # [B, S, F]

        # ------- 构造布尔 padding mask -------
        # True=屏蔽；False=参与注意力
        src_key_padding_mask = torch.ones((B, S), dtype=torch.bool, device=device)
        # weight(0) 与 self(1) token 永远激活
        src_key_padding_mask[:, 0] = False
        src_key_padding_mask[:, 1] = False

        if self.task_obs_onehot_size > 0:
            # onehot 区段：位于原 task_obs 的末尾
            # 形状应为 [B, onehot_size]
            onehot = task_obs[..., self.task_obs_tota_size:]
            assert onehot.shape == (B, self.task_obs_onehot_size), \
                f"onehot shape {tuple(onehot.shape)} != (B, onehot_size)=({B}, {self.task_obs_onehot_size})"

            # 选中当前子任务的 index（2 起步，因为 0/1 已被占用）
            task_idx_in_seq = onehot.argmax(dim=-1).to(torch.long) + 2  # [B]
            # 安全范围：2..S-1
            assert (task_idx_in_seq >= 2).all() and (task_idx_in_seq < S).all(), \
                f"task token index out of range (2..{S-1})"

            # 将所选子任务 token 置为激活（False）
            src_key_padding_mask[torch.arange(B, device=device), task_idx_in_seq] = False

        # ------- 关键安全检查：mask/shape/device/有效长度 -------
        _assert_bool_mask(src_key_padding_mask, B, S, x.device)

        # ------- 调用 Transformer（强制 math 路径以避坑） -------
        try:
            from torch.backends.cuda import sdp_kernel
            with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, S, F]
        except Exception:
            # 低版本 PyTorch 没有 sdp_kernel 上下文就直接调用
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 取 weight token 输出
        output = self.composer(x[:, 0])  # [B, act_dim]
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
            mean = self._eval_multitask_transformer(observations)
        else:
            # DecisionTransformer 分支
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
            actions_mean = self._eval_multitask_transformer(observations)
        else:
            actions_mean = self.actor(
                cur_timestep, observations[:, :, :36], observations[:, :, 36:48]
            )
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        if len(critic_observations.shape) == 3:
            critic_observations = critic_observations[:, -1, :]
        value = self.critic(critic_observations)
        return value

    def eval_actor(self, obs, cur_timestep=None):
        if self.enable_multi_task and self.actor is None:
            mu = self._eval_multitask_transformer(obs)
        else:
            mu = self.actor(cur_timestep, obs[:, :, :36], obs[:, :, 36:48])
        sigma = self.std
        return mu, sigma

    def eval_critic(self, obs):
        if len(obs.shape) == 3:
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
        return nn.ELU()  # 给一个安全的默认激活，KISS
