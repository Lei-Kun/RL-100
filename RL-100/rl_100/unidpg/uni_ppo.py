import gym
import torch
import time
import numpy as np
import csv
import inspect
from rl_100.unidpg.buffer import OnlineReplayBuffer, OfflineReplayBuffer
from rl_100.unidpg.net import GaussPolicyMLP
from rl_100.unidpg.critic import ValueLearner, QLearner, IQL_Q_V
from rl_100.unidpg.ppo import ProximalPolicyOptimization
from rl_100.unidpg.utils import CONST_EPS, log_prob_func, orthogonal_initWeights, AdaptiveScheduler
import os
from copy import deepcopy
from torch.distributions.categorical import Categorical
from rl_100.policy.rl100_3d import RL1003D
import pdb
from rl_100.unidpg.diffusion_policy.helpers import Losses
from rl_100.unidpg.diffusion_policy.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
from rl_100.unidpg.transition_model.dynamics import EnsembleDynamics
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn   
import gym
from rl_100.common.pytorch_util import dict_apply
import os
from termcolor import cprint
from rl_100.model.common.cm_util import update_ema
from collections import defaultdict
from contextlib import contextmanager
from omegaconf import OmegaConf, DictConfig


def compute_gae_per_env(rewards, dones, dws, vs, vs_, gamma, lamda, n_action_steps=1):
    """
    Per-env GAE computation that avoids cross-env propagation.

    Args:
        rewards: (steps, env_num, 1)
        dones:   (steps, env_num, 1)
        dws:     (steps, env_num, 1) — true termination (not truncation)
        vs:      (steps, env_num, 1) — V(s)
        vs_:     (steps, env_num, 1) — V(s')
        gamma:   discount factor
        lamda:   GAE lambda
        n_action_steps: action chunk size

    Returns:
        adv:      (steps * env_num, 1)
        v_target: (steps * env_num, 1)
    """
    steps, env_num = rewards.shape[0], rewards.shape[1]
    gamma_eff = gamma ** n_action_steps
    deltas = rewards + gamma_eff * (1.0 - dws) * vs_ - vs  # (steps, env_num, 1)

    adv = torch.zeros_like(rewards)  # (steps, env_num, 1)
    gae = torch.zeros(env_num, 1, device=rewards.device)  # per-env accumulator

    for t in reversed(range(steps)):
        gae = deltas[t] + gamma_eff * lamda * gae * (1.0 - dones[t])
        adv[t] = gae

    v_target = adv + vs
    return adv.reshape(-1, 1), v_target.reshape(-1, 1)


class BehaviorProximalPolicyOptimization(ProximalPolicyOptimization):

    def __init__(
        self,
        policy: RL1003D,
        device: torch.device,
        policy_lr: float,
        clip_ratio: float,
        entropy_weight: float,
        decay: float,
        omega: float,
        batch_size: int,
        is_iql: bool,
        temperature: float = None,
        ratio_strategy: str = 'last',
        top_k: int = 1,
        num_inference_steps: int = 5,
        truncated_backprop: bool = False,
        truncated_backprop_timestep: int = 0,
        fix_encoder: bool = False,
        cfg: dict = None,
    ) -> None:
        super().__init__(
            policy = policy,
            device = device,
            policy_lr = policy_lr,
            clip_ratio = clip_ratio,
            entropy_weight = entropy_weight,
            decay = decay,
            omega = omega,
            batch_size = batch_size,
            is_iql=is_iql,
            ratio_strategy=ratio_strategy,
            fix_encoder = fix_encoder)

        self.temperature = temperature
        if top_k == 1:
            top_k = num_inference_steps
        self.num_inference_steps = num_inference_steps
        self.top_k = top_k
        self.bc_loss = Losses['l2']()
        self.truncated_backprop = truncated_backprop
        self.truncated_backprop_timestep = truncated_backprop_timestep
        self.cfg = cfg
        self.iteration = 0
        # ---- lighting-aug defaults (overridden in transfer2online); off on every path ----
        self.freeze_rgb_backbone = False
        self.aux_enabled = False
        self.aux_cfg = {}
        self.aux_step = 0
        self.last_aux_metrics = {}
        self._aux_metric_buffer = defaultdict(list)
        self.aux_photometric = None
        self.aux_generator = None

        if truncated_backprop_timestep == 0:
            self.truncated_backprop_timestep = num_inference_steps - 1

        # Ratio statistics monitoring
        ppo_cfg = getattr(cfg, "ppo", None)
        self._enable_ratio_logging = bool(getattr(ppo_cfg, "enable_ratio_logging", False))
        self._ratio_log_every_updates = max(1, int(getattr(ppo_cfg, "ratio_log_every_updates", 10)))
        self._ratio_plot_on_final_flush = bool(getattr(ppo_cfg, "ratio_plot_on_final_flush", True))
        self._ratio_records = []
        self._delta_cache = []
        self._ratio_log_counter = 0
        self._ratio_log_dir = None
        self._ratio_log_flush_interval = 50
        self._ratio_log_written_until = 0
        self._debug_stats_records = []
    
    def _compute_critic_values_in_chunks(self, s, s_, use_obs2latent=True):
        """
        分批计算critic值以避免显存溢出
        
        Args:
            s: 当前状态
            s_: 下一状态
            use_obs2latent: 是否使用obs2latent而不是obs2this_nobs
            
        Returns:
            vs, vs_: 当前状态和下一状态的critic值
        """
        batch_size = s['agent_pos'].shape[0]
        configured_chunk_size = int(
            getattr(self.args, 'critic_value_chunk_size', 256))
        chunk_size = min(max(1, configured_chunk_size), batch_size)
        num_chunks = (batch_size + chunk_size - 1) // chunk_size
        vs_list, vs_list_ = [], []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, batch_size)
            
            s_chunk = dict_apply(
                s, lambda x: x[start_idx:end_idx].to(self._device))
            s_chunk_ = dict_apply(
                s_, lambda x: x[start_idx:end_idx].to(self._device))
            
            if use_obs2latent:
                critic_s_chunk = self._policy.obs2latent(s_chunk)
                critic_s_chunk_ = self._policy.obs2latent(s_chunk_)
            else:
                critic_s_chunk = self._policy.obs2this_nobs(s_chunk)
                critic_s_chunk_ = self._policy.obs2this_nobs(s_chunk_)
            
            vs_chunk = self.critic(critic_s_chunk)
            vs_chunk_ = self.critic(critic_s_chunk_)
            
            vs_list.append(vs_chunk)
            vs_list_.append(vs_chunk_)
        
        # 合并所有chunk的结果
        vs = torch.cat(vs_list, dim=0)
        vs_ = torch.cat(vs_list_, dim=0)
        
        return vs, vs_
        
    @torch.no_grad()
    def advantage_computation(
        self,
        s: torch.Tensor,
        action: torch.Tensor,
        value: ValueLearner, 
        Q: QLearner = None,
        iql: IQL_Q_V = None) -> torch.Tensor:
        if self._is_iql:
            if hasattr(self._policy, 'diffusion_eta'):
                if self._policy.diffusion_eta:
                    action = action[..., :-1]
                    
            advantage = iql.get_advantage(s, action)
            if self.temperature:
                print('using advantage with exp temperature')
                advantage = torch.minimum(torch.exp(advantage * self.temperature), torch.ones_like(advantage).to(self._device)*100.0)
            advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
        else:
            s_value = value(s)
            if isinstance(Q, list): # fitted q_pi else q_bc
                advantage = Q(s, action) - s_value
            else:
                advantage = Q(s, action) - s_value
            advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
            advantage = self.weighted_advantage(advantage)

        return advantage

    def _compute_advantage_actor_only(self, s, action, value, Q, iql):
        """Compute advantage without building critic gradient graph (saves VRAM)."""
        with torch.no_grad():
            advantage = self.advantage_computation(s, action, value, Q, iql)
        return advantage

    @torch.no_grad()
    def _evaluate_value_function(self, state_features, value, iql):
        if iql is not None:
            return iql._value(state_features)
        if value is not None:
            return value(state_features)
        raise ValueError("Both value and iql are None; cannot evaluate value function.")

    def set_ratio_log_dir(self, log_dir):
        """Set directory for ratio statistics CSV and plots."""
        if not self._enable_ratio_logging:
            self._ratio_log_dir = None
            return
        self._ratio_log_dir = log_dir
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

    def _record_ratio_stats(self, phase, denoise_step, ratio, old_logprob, new_logprob):
        """Record PPO ratio statistics per denoise step for debugging."""
        if not self._enable_ratio_logging:
            return
        if self._ratio_log_every_updates > 1 and (self.iteration % self._ratio_log_every_updates) != 0:
            return
        ratio_flat = ratio.detach().float().reshape(-1).cpu()
        old_flat = old_logprob.detach().float().reshape(-1).cpu()
        new_flat = new_logprob.detach().float().reshape(-1).cpu()
        delta_flat = new_flat - old_flat

        record = {
            "idx": self._ratio_log_counter,
            "phase": phase,
            "denoise_step": denoise_step,
            "ratio_mean": float(ratio_flat.mean()),
            "ratio_q05": float(torch.quantile(ratio_flat, 0.05)),
            "ratio_q25": float(torch.quantile(ratio_flat, 0.25)),
            "ratio_q50": float(torch.quantile(ratio_flat, 0.50)),
            "ratio_q75": float(torch.quantile(ratio_flat, 0.75)),
            "ratio_q95": float(torch.quantile(ratio_flat, 0.95)),
            "old_logprob_mean": float(old_flat.mean()),
            "new_logprob_mean": float(new_flat.mean()),
            "delta_mean": float(delta_flat.mean()),
            "delta_std": float(delta_flat.std(unbiased=False)),
            "delta_q05": float(torch.quantile(delta_flat, 0.05)),
            "delta_q25": float(torch.quantile(delta_flat, 0.25)),
            "delta_q50": float(torch.quantile(delta_flat, 0.50)),
            "delta_q75": float(torch.quantile(delta_flat, 0.75)),
            "delta_q95": float(torch.quantile(delta_flat, 0.95)),
        }
        self._ratio_records.append(record)
        self._delta_cache.append(delta_flat.numpy())
        if len(self._delta_cache) > 200:
            self._delta_cache = self._delta_cache[-200:]

        self._ratio_log_counter += 1
        if self._ratio_log_dir is not None and (self._ratio_log_counter % self._ratio_log_flush_interval == 0):
            self.flush_ratio_logs(force=False)

    def _record_debug_stats(self, denoise_step, debug, next_sample):
        """Record per-step std/SNR stats from flow scheduler debug output."""
        if not self._enable_ratio_logging:
            return
        if self._ratio_log_every_updates > 1 and (self.iteration % self._ratio_log_every_updates) != 0:
            return
        mean = debug['mean'].detach().float()
        std = debug['std'].detach().float()
        ns = next_sample.detach().float()
        residual = (ns - mean).norm(dim=-1).mean()
        std_mean = std.clamp(min=1e-12).mean()
        snr = residual / std_mean

        record = {
            "idx": self._ratio_log_counter,
            "denoise_step": denoise_step,
            "std_mean": float(std_mean),
            "mean_norm": float(mean.norm(dim=-1).mean()),
            "residual_norm": float(residual),
            "snr": float(snr),
        }
        self._debug_stats_records.append(record)

    def _step_forward_logprob_with_optional_debug(
        self,
        model_output,
        timesteps,
        sample,
        next_sample,
        denoise_step,
        eta=1.0,
    ):
        """Call scheduler step_forward_logprob, optionally collecting debug stats.

        Keeps default scheduler behavior unchanged unless ratio logging is enabled
        and the scheduler explicitly supports `return_debug`.
        """
        scheduler = self._policy.noise_scheduler
        want_debug = self._enable_ratio_logging
        supports_debug = False
        if want_debug:
            try:
                supports_debug = "return_debug" in inspect.signature(
                    scheduler.step_forward_logprob
                ).parameters
            except (TypeError, ValueError):
                supports_debug = False

        kwargs = {
            "next_sample": next_sample,
            "eta": eta,
        }
        if getattr(self._policy, "is_flow", False):
            kwargs["step_index"] = denoise_step
        if want_debug and supports_debug:
            kwargs["return_debug"] = True
            log_prob, debug = scheduler.step_forward_logprob(
                model_output, timesteps, sample, **kwargs
            )
            self._record_debug_stats(denoise_step, debug, next_sample)
            return log_prob

        return scheduler.step_forward_logprob(
            model_output, timesteps, sample, **kwargs
        )

    def flush_ratio_logs(self, force=True):
        """Write ratio statistics to CSV and generate plots."""
        if not self._enable_ratio_logging or self._ratio_log_dir is None:
            return

        # Always flush debug_stats when force=True, regardless of ratio_records
        if force:
            debug_records = getattr(self, '_debug_stats_records', [])
            if debug_records:
                debug_csv = os.path.join(self._ratio_log_dir, "debug_stats.csv")
                headers = list(debug_records[0].keys())
                with open(debug_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(debug_records)
                self._debug_stats_records = []

        if len(self._ratio_records) == 0:
            return
        if not force and len(self._ratio_records) < self._ratio_log_flush_interval:
            return

        csv_path = os.path.join(self._ratio_log_dir, "ratio_stats.csv")
        pending_records = self._ratio_records[self._ratio_log_written_until :]
        if pending_records:
            headers = list(pending_records[0].keys())
            file_exists = os.path.exists(csv_path)
            mode = "a" if file_exists and self._ratio_log_written_until > 0 else "w"
            with open(csv_path, mode, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if mode == "w":
                    writer.writeheader()
                writer.writerows(pending_records)
            self._ratio_log_written_until = len(self._ratio_records)

        # Prune written records to bound memory and prevent GC pressure growth
        if not force:
            self._ratio_records = self._ratio_records[self._ratio_log_written_until:]
            self._ratio_log_written_until = 0
            return

        if not self._ratio_plot_on_final_flush:
            self._ratio_records = self._ratio_records[self._ratio_log_written_until:]
            self._ratio_log_written_until = 0
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = [r["idx"] for r in self._ratio_records]
            plt.figure(figsize=(10, 4))
            plt.plot(xs, [r["ratio_mean"] for r in self._ratio_records], label="ratio_mean")
            plt.plot(xs, [r["ratio_q05"] for r in self._ratio_records], label="ratio_q05", alpha=0.7)
            plt.plot(xs, [r["ratio_q50"] for r in self._ratio_records], label="ratio_q50", alpha=0.9)
            plt.plot(xs, [r["ratio_q95"] for r in self._ratio_records], label="ratio_q95", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self._ratio_log_dir, "ratio_quantiles.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(10, 4))
            plt.plot(xs, [r["delta_mean"] for r in self._ratio_records], label="delta_mean")
            plt.plot(xs, [r["delta_q05"] for r in self._ratio_records], label="delta_q05", alpha=0.7)
            plt.plot(xs, [r["delta_q95"] for r in self._ratio_records], label="delta_q95", alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self._ratio_log_dir, "delta_quantiles.png"), dpi=150)
            plt.close()
        except Exception:
            pass

        # Prune after final flush (plots already consumed the data)
        self._ratio_records.clear()
        self._ratio_log_written_until = 0

    @torch.no_grad()
    def NStepValueEstimation(self, nobs_features, nactions, dynamics, value, Q, iql, opt_steps):
        advantages, rewards = [], []
        # dp predicts n steps action, compute advantage value for each step
        for i in range(opt_steps):
            advantage = self.advantage_computation(nobs_features, nactions[:, i], value, Q, iql)
            advantages.append(advantage)
            next_nobs_features, reward, _, _ = dynamics.step(nobs_features, nactions[:, i])
            rewards.append(torch.from_numpy(reward).to(self._device))
            nobs_features = torch.from_numpy(next_nobs_features).to(self._device)
        return torch.stack(advantages), torch.stack(rewards)
    @torch.no_grad()  
    def GAE_withQ(self, advantages, gamma, lamda, dones=None):
        gae_advantages, gae = [], 0
        deltas = advantages # advantages
        if dones is None:
            dones = torch.zeros_like(deltas)
        for delta, done in zip(reversed(deltas), reversed(dones)):
            gae = delta + gamma * lamda * gae * (1.0 - done)
            gae_advantages.insert(0, gae)
        return torch.stack(gae_advantages)

    def _get_chunk_action_bounds(self, opt_steps):
        action_start = 0 if self.cfg.no_pre_action else self.n_obs_steps - 1
        action_end = action_start + opt_steps
        return action_start, action_end

    def _slice_chunk_actions(self, actions, opt_steps):
        action_start, action_end = self._get_chunk_action_bounds(opt_steps)
        return actions[:, action_start:action_end]

    @staticmethod
    def _sum_logprob_event_dims(logprob):
        """Return one joint log-prob scalar per batch item."""
        if logprob.ndim <= 1:
            return logprob.reshape(logprob.shape[0])
        return logprob.reshape(logprob.shape[0], -1).sum(dim=1)

    @staticmethod
    def _sum_step_logprob_event_dims(logprob):
        """Return one joint log-prob scalar per batch item and action step."""
        if logprob.ndim <= 2:
            return logprob
        return logprob.reshape(logprob.shape[0], logprob.shape[1], -1).sum(dim=2)

    def _current_policy_obs2feature(self, obs, fix_encoder, training):
        kwargs = {"fix_encoder": fix_encoder}
        if "training" in inspect.signature(self._policy.obs2feature).parameters:
            kwargs["training"] = training
        return self._policy.obs2feature(obs, **kwargs)

    def _get_offline_chunk_modes(self):
        ratio_mode = getattr(self.cfg, 'offline_chunk_ratio_mode', 'scalar')
        adv_mode = getattr(self.cfg, 'offline_chunk_adv_mode', 'scalar_iql')
        valid_ratio_modes = {'per_step', 'scalar'}
        valid_adv_modes = {'scalar_iql', 'per_step_vdelta', 'chunk_vdelta_scalar', 'chunk_vdelta_gae'}

        if ratio_mode not in valid_ratio_modes:
            raise ValueError(f"Unsupported offline_chunk_ratio_mode={ratio_mode}")
        if adv_mode not in valid_adv_modes:
            raise ValueError(f"Unsupported offline_chunk_adv_mode={adv_mode}")
        if ratio_mode == 'scalar' and adv_mode not in ('scalar_iql', 'chunk_vdelta_scalar', 'chunk_vdelta_gae'):
            raise ValueError(
                f"offline_chunk_ratio_mode={ratio_mode} requires "
                f"offline_chunk_adv_mode in (scalar_iql, chunk_vdelta_scalar, chunk_vdelta_gae), got {adv_mode}"
            )
        if adv_mode == 'chunk_vdelta_gae' and ratio_mode != 'scalar':
            raise ValueError(
                f"offline_chunk_adv_mode=chunk_vdelta_gae requires "
                f"offline_chunk_ratio_mode=scalar, got {ratio_mode}"
            )
        if adv_mode == 'per_step_vdelta':
            raise ValueError(
                "offline_chunk_adv_mode=per_step_vdelta is invalid with "
                "chunk_as_single_action=True because chunk dynamics expects "
                "the whole action chunk, not single-step actions"
            )
        return ratio_mode, adv_mode

    def _apply_chunk_adv_clip(self, advantages):
        chunk_adv_clip = getattr(self.cfg, 'chunk_adv_clip', None)
        if chunk_adv_clip is None:
            return advantages
        return torch.clamp(advantages, -chunk_adv_clip, chunk_adv_clip)

    @torch.no_grad()
    def _compute_chunk_step_advantages_vdelta(
        self,
        nobs_features,
        chunk_actions,
        dynamics,
        value,
        iql,
        gamma,
        lamda,
        use_gae,
    ):
        if dynamics is None:
            raise ValueError("dynamics is required for offline_chunk_adv_mode=per_step_vdelta")
        if not getattr(dynamics, 'predict_r', False):
            raise ValueError(
                "per_step_vdelta requires predict_r=True in dynamics config; "
                "otherwise rollout returns zero rewards and vdelta degenerates"
            )

        batch_size = nobs_features.shape[0]
        feature_dim = nobs_features.shape[1] // self.cfg.n_obs_steps
        policy_features = nobs_features.reshape(batch_size, self.cfg.n_obs_steps, feature_dim)
        single_nob_features = policy_features[:, -1, :]
        advantages = []
        terminals = []

        for step_idx in range(chunk_actions.shape[1]):
            state_features = policy_features.reshape(batch_size, -1)
            value_now = self._evaluate_value_function(state_features, value, iql).reshape(batch_size, -1)[:, 0]

            # Match dynamics.multi_step()/multi_step_evaluation() semantics:
            # pass the last-frame feature as the primary state input, and the
            # full observation window separately when prediction_mode="full".
            next_obs, reward, terminal, _ = dynamics.step(
                single_nob_features,
                chunk_actions[:, step_idx],
                policy_features,
            )

            reward_t = torch.from_numpy(reward).to(
                device=self._device,
                dtype=value_now.dtype,
            ).reshape(batch_size, -1)[:, 0]
            terminal_t = torch.from_numpy(terminal).to(
                device=self._device,
                dtype=value_now.dtype,
            ).reshape(batch_size, -1)[:, 0]

            if dynamics.prediction_mode == "full":
                next_policy_features = torch.from_numpy(next_obs).to(
                    device=self._device,
                    dtype=policy_features.dtype,
                )
                next_single_nob_features = next_policy_features[:, -1, :]
            else:
                next_single_nob_features = torch.from_numpy(next_obs).to(
                    device=self._device,
                    dtype=policy_features.dtype,
                )
                next_policy_features = torch.cat(
                    (policy_features[:, 1:, :], next_single_nob_features.unsqueeze(1)),
                    dim=1,
                )

            next_state_features = next_policy_features.reshape(batch_size, -1)
            value_next = self._evaluate_value_function(next_state_features, value, iql).reshape(batch_size, -1)[:, 0]
            delta = reward_t + gamma * (1 - terminal_t) * value_next - value_now
            advantages.append(delta)
            terminals.append(terminal_t)

            policy_features = next_policy_features
            single_nob_features = next_single_nob_features

        advantages = torch.stack(advantages)
        if use_gae:
            advantages = self.GAE_withQ(advantages, gamma, lamda, dones=torch.stack(terminals))
        # Normalize per-step advantages across batch (matches scalar_iql normalization)
        for j in range(advantages.shape[0]):
            adv_j = advantages[j]
            advantages[j] = (adv_j - adv_j.mean()) / (adv_j.std() + CONST_EPS)
        return advantages

    @torch.no_grad()
    def _compute_chunk_scalar_advantage_vdelta(
        self,
        nobs_features,
        chunk_actions,
        dynamics,
        value,
        iql,
        gamma,
    ):
        if dynamics is None:
            raise ValueError("dynamics is required for offline_chunk_adv_mode=chunk_vdelta_scalar")

        batch_size = nobs_features.shape[0]
        feature_dim = nobs_features.shape[1] // self.cfg.n_obs_steps
        policy_features = nobs_features.reshape(batch_size, self.cfg.n_obs_steps, feature_dim)

        result = self._compute_single_chunk_boundary_delta(
            policy_features, chunk_actions, dynamics, value, iql, gamma,
        )
        advantages = result["delta"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + CONST_EPS)
        return advantages

    @torch.no_grad()
    def _compute_single_chunk_boundary_delta(
        self,
        policy_features,
        chunk_action,
        dynamics,
        value,
        iql,
        gamma,
    ):
        if not getattr(dynamics, 'predict_r', False):
            raise ValueError(
                "chunk boundary delta requires predict_r=True in dynamics config"
            )
        batch_size = policy_features.shape[0]
        single_nob_features = policy_features[:, -1, :]
        state_features = policy_features.reshape(batch_size, -1)
        value_now = self._evaluate_value_function(state_features, value, iql).reshape(batch_size, -1)[:, 0]

        next_obs, reward, terminal, _ = dynamics.step(
            single_nob_features,
            chunk_action,
            policy_features,
        )

        reward_t = torch.from_numpy(reward).to(
            device=self._device, dtype=value_now.dtype,
        ).reshape(batch_size, -1)[:, 0]
        terminal_t = torch.from_numpy(terminal).to(
            device=self._device, dtype=value_now.dtype,
        ).reshape(batch_size, -1)[:, 0]

        if dynamics.prediction_mode == "full":
            next_policy_features = torch.from_numpy(next_obs).to(
                device=self._device, dtype=policy_features.dtype,
            )
        else:
            next_single = torch.from_numpy(next_obs).to(
                device=self._device, dtype=policy_features.dtype,
            )
            next_policy_features = torch.cat(
                (policy_features[:, 1:, :], next_single.unsqueeze(1)), dim=1,
            )

        next_state_features = next_policy_features.reshape(batch_size, -1)
        value_next = self._evaluate_value_function(next_state_features, value, iql).reshape(batch_size, -1)[:, 0]

        K = chunk_action.shape[1] if chunk_action.ndim > 1 else 1
        gamma_K = torch.as_tensor(gamma, device=self._device, dtype=value_now.dtype) ** K
        delta = reward_t + gamma_K * (1 - terminal_t) * value_next - value_now

        return {
            "delta": delta,
            "reward": reward_t,
            "terminal": terminal_t,
            "value_now": value_now,
            "value_next": value_next,
            "next_policy_features": next_policy_features,
        }

    @torch.no_grad()
    def _compute_chunk_gae_advantage_vdelta(
        self,
        nobs_features,
        chunk_actions,
        dynamics,
        value,
        iql,
        gamma,
        gae_lambda,
        n_rollout,
        chunk_source,
    ):
        if dynamics is None:
            raise ValueError("dynamics is required for offline_chunk_adv_mode=chunk_vdelta_gae")
        if not getattr(dynamics, 'predict_r', False):
            raise ValueError(
                "chunk_vdelta_gae requires predict_r=True in dynamics config; "
                "otherwise rollout returns zero rewards and vdelta degenerates"
            )
        if chunk_source != 'repeat_first':
            raise ValueError(
                f"chunk_vdelta_gae v1 only supports chunk_source='repeat_first', got '{chunk_source}'"
            )

        batch_size = nobs_features.shape[0]
        feature_dim = nobs_features.shape[1] // self.cfg.n_obs_steps
        policy_features = nobs_features.reshape(batch_size, self.cfg.n_obs_steps, feature_dim)

        K = chunk_actions.shape[1] if chunk_actions.ndim > 1 else 1
        gamma_K = torch.as_tensor(gamma, device=self._device, dtype=policy_features.dtype) ** K

        deltas = []
        terminals = []
        alive_mask = torch.ones(batch_size, device=self._device, dtype=policy_features.dtype)
        current_features = policy_features

        for step_idx in range(n_rollout):
            # Early exit if all samples terminated
            if alive_mask.sum() == 0:
                remaining = n_rollout - len(deltas)
                for _ in range(remaining):
                    deltas.append(torch.zeros(batch_size, device=self._device, dtype=policy_features.dtype))
                    terminals.append(torch.ones(batch_size, device=self._device, dtype=policy_features.dtype))
                break

            result = self._compute_single_chunk_boundary_delta(
                current_features, chunk_actions, dynamics, value, iql, gamma,
            )
            delta = result["delta"]
            terminal_t = result["terminal"]
            next_features = result["next_policy_features"]

            deltas.append(delta * alive_mask)
            terminals.append(terminal_t)

            alive_mask = alive_mask * (1.0 - terminal_t)
            mask = alive_mask[:, None, None]
            current_features = mask * next_features + (1.0 - mask) * current_features

        deltas = torch.stack(deltas)  # (n_rollout, B)
        terminals = torch.stack(terminals)  # (n_rollout, B)

        # For GAE: done[k] should mask contribution from step k+1
        # terminals[k] = whether transition k hits terminal, which is what we need
        # No shift needed — terminals[k] directly masks gae propagation from k+1 to k
        gae_adv = self.GAE_withQ(deltas, gamma_K, gae_lambda, dones=terminals)
        advantages = gae_adv[0]  # first chunk's GAE advantage

        advantages = (advantages - advantages.mean()) / (advantages.std() + CONST_EPS)
        return advantages

    def update_distribution(
        self, 
        batch: dict,
        value: ValueLearner, 
        Q: QLearner,
        iql: IQL_Q_V,
        is_clip_decay: bool,
        is_lr_decay: bool,
        is_linear_decay: bool =  None,
        bppo_lr_now: float =  None, 
        clip_ratio_now: float =  None,
        is_bc_loss: bool = False,
        final_reward: bool = True,
        dynamics: EnsembleDynamics = None,
        use_gae: bool = True,
        gamma: float = 0.99,
        lamda: float =0.95,
        fix_encoder: bool = True
    ) -> float:

        horizon = (
            self.cfg.horizon - (self.cfg.n_obs_steps - 1)
            if self.cfg.no_pre_action
            else self.cfg.horizon
        )
        if self.cfg.ft_all_actions:
            opt_steps = horizon
        else:
            opt_steps = self.cfg.n_action_steps
        chunk_ratio_mode = 'per_step'
        chunk_adv_mode = 'scalar_iql'
        if self.cfg.chunk_as_single_action and opt_steps > 1:
            chunk_ratio_mode, chunk_adv_mode = self._get_offline_chunk_modes()
        # nobs = self.normalizer.normalize(batch['obs'])
        if fix_encoder:
            old_all_x, old_all_next_x, old_all_logprob, cond_data, cond_mask, local_cond, global_cond, nobs_features = self._old_policy.all_step_logprob(batch['obs'], fix_encoder=fix_encoder, training=True) # (num_inference_steps, batch_size, dim)
        else:
            with torch.no_grad():
                old_all_x, old_all_next_x, old_all_logprob, cond_data, cond_mask, local_cond, global_cond, nobs_features = self._old_policy.all_step_logprob(batch['obs'], fix_encoder=fix_encoder, training=True) # (num_inference_steps, batch_size, dim)
            cond_data, cond_mask, local_cond, global_cond, nobs_features = self._current_policy_obs2feature(
                batch['obs'],
                fix_encoder=False,
                training=True,
            )
        # x - shape: (batch size, n_step_actions, action_dim)

        # old_all_x, old_all_logprob, new_all_logprob = reversed(old_all_x), reversed(old_all_logprob),  reversed(new_all_logprob)
        losses = []
        use_chunk_level_ratio = getattr(self.cfg, 'bppo_chunk_level_ratio', True)
        if final_reward:
                if opt_steps >1:
                    if self.cfg.chunk_as_single_action:
                        chunk_actions = self._slice_chunk_actions(old_all_next_x[-1], opt_steps)
                        if chunk_adv_mode == 'scalar_iql':
                            advantages = self._compute_advantage_actor_only(nobs_features, chunk_actions, value, Q, iql)
                        elif chunk_adv_mode == 'per_step_vdelta':
                            advantages = self._compute_chunk_step_advantages_vdelta(
                                nobs_features,
                                chunk_actions,
                                dynamics,
                                value,
                                iql,
                                gamma,
                                lamda,
                                use_gae,
                            )
                        elif chunk_adv_mode == 'chunk_vdelta_scalar':
                            advantages = self._compute_chunk_scalar_advantage_vdelta(
                                nobs_features,
                                chunk_actions,
                                dynamics,
                                value,
                                iql,
                                gamma,
                            )
                        elif chunk_adv_mode == 'chunk_vdelta_gae':
                            n_rollout = int(getattr(self.cfg, 'chunk_vdelta_gae_n_rollout', 3))
                            gae_lambda = float(getattr(self.cfg, 'chunk_vdelta_gae_lambda', 0.95))
                            chunk_source = str(getattr(self.cfg, 'chunk_vdelta_gae_chunk_source', 'repeat_first'))
                            advantages = self._compute_chunk_gae_advantage_vdelta(
                                nobs_features,
                                chunk_actions,
                                dynamics,
                                value,
                                iql,
                                gamma,
                                gae_lambda,
                                n_rollout,
                                chunk_source,
                            )
                        else:
                            raise ValueError(f"Unsupported offline_chunk_adv_mode={chunk_adv_mode}")
                        advantages = self._apply_chunk_adv_clip(advantages)
                    else:
                        advantages, rewards = self.NStepValueEstimation(nobs_features, old_all_next_x[-1], dynamics, value, Q, iql, opt_steps) # n_step_actions, batch_size
                        if use_gae:
                            advantages = self.GAE_withQ(advantages, gamma, lamda)
                else:
                    if self.cfg.no_pre_action:
                        advantages = self._compute_advantage_actor_only(nobs_features, old_all_next_x[-1][:, 0], value, Q, iql)
                    else:
                        advantages = self._compute_advantage_actor_only(nobs_features, old_all_next_x[-1][:, self.n_obs_steps - 1], value, Q, iql)

        for i, t in enumerate(self._old_policy.noise_scheduler.timesteps):
                timesteps = t
                if not torch.is_tensor(timesteps):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    timesteps = torch.tensor([timesteps], dtype=torch.long, device=self._device)
                elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(self._device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(global_cond.shape[0])
                unet_timesteps = self._policy.get_unet_timesteps(timesteps)

                model_output = self._policy.model(sample=old_all_x[i],
                                timestep=unet_timesteps,
                                local_cond=local_cond, global_cond=global_cond)
                if hasattr(self._policy, 'diffusion_eta'):
                    if self._policy.diffusion_eta:
                        eta = torch.mean(model_output[:, :, -1], dim=1, keepdim=True)
                else:   
                    eta = 1.
                    
                if is_bc_loss:
                    bc_loss = self.bc_loss(model_output, old_all_x[0], 1.0)
                new_log_prob = self._step_forward_logprob_with_optional_debug(
                    model_output=model_output,
                    timesteps=timesteps,
                    sample=old_all_x[i],
                    next_sample=old_all_next_x[i],
                    denoise_step=i,
                    eta=eta,
                )

                # old_log_prob, new_log_prob = old_log_prob.squeeze(1), new_log_prob.squeeze(1)
                if not final_reward:
                    if opt_steps >1:
                        if self.cfg.chunk_as_single_action:
                            chunk_actions = self._slice_chunk_actions(old_all_next_x[i], opt_steps)
                            if chunk_adv_mode == 'scalar_iql':
                                advantages = self.advantage_computation(nobs_features, chunk_actions, value, Q, iql)
                            elif chunk_adv_mode == 'per_step_vdelta':
                                advantages = self._compute_chunk_step_advantages_vdelta(
                                    nobs_features,
                                    chunk_actions,
                                    dynamics,
                                    value,
                                    iql,
                                    gamma,
                                    lamda,
                                    use_gae,
                                )
                            elif chunk_adv_mode == 'chunk_vdelta_scalar':
                                advantages = self._compute_chunk_scalar_advantage_vdelta(
                                    nobs_features,
                                    chunk_actions,
                                    dynamics,
                                    value,
                                    iql,
                                    gamma,
                                )
                            elif chunk_adv_mode == 'chunk_vdelta_gae':
                                raise ValueError(
                                    "chunk_vdelta_gae only supports final_reward=True"
                                )
                            else:
                                raise ValueError(f"Unsupported offline_chunk_adv_mode={chunk_adv_mode}")
                            advantages = self._apply_chunk_adv_clip(advantages)
                        else:
                            advantages, rewards = self.NStepValueEstimation(nobs_features, old_all_next_x[i], dynamics, value, Q, iql, opt_steps) # n_step_actions, batch_size
                            if use_gae:
                                advantages = self.GAE_withQ(advantages, gamma, lamda)
                    else:
                        if self.cfg.no_pre_action:
                            advantages = self._compute_advantage_actor_only(nobs_features, old_all_next_x[i][:, 0], value, Q, iql)
                        else:
                            advantages = self._compute_advantage_actor_only(nobs_features, old_all_next_x[i][:, self.n_obs_steps - 1], value, Q, iql)
            
                if is_clip_decay:
                    if is_linear_decay:
                        self._clip_ratio = clip_ratio_now
                    else:
                        self._clip_ratio = self._clip_ratio * self._decay

                if self.cfg.chunk_as_single_action and use_chunk_level_ratio:
                    action_start, action_end = self._get_chunk_action_bounds(opt_steps)
                    logprob_old = old_all_logprob[i][:, action_start:action_end]
                    logprob_new = new_log_prob[:, action_start:action_end]
                    if chunk_ratio_mode == 'scalar':
                        old_logprob_scalar = logprob_old.reshape(logprob_old.shape[0], -1).sum(dim=1)
                        new_logprob_scalar = logprob_new.reshape(logprob_new.shape[0], -1).sum(dim=1)
                        ratio_scalar = (new_logprob_scalar - old_logprob_scalar).exp()
                        self._record_ratio_stats(
                            "offline_multi",
                            i,
                            ratio_scalar,
                            old_logprob_scalar,
                            new_logprob_scalar,
                        )

                        adv = advantages.detach().reshape(advantages.shape[0], -1)
                        assert adv.shape[1] == 1, \
                            f"chunk_as_single_action expects scalar advantage, got {adv.shape}"
                        adv = adv[:, 0]

                        loss1 = ratio_scalar * adv
                        loss2 = torch.clamp(ratio_scalar, 1 - self._clip_ratio, 1 + self._clip_ratio) * adv
                        loss = -(torch.min(loss1, loss2)).mean()
                    else:
                        ratio_perstep = (logprob_new.sum(-1) - logprob_old.sum(-1)).exp()
                        self._record_ratio_stats(
                            "offline_multi",
                            i,
                            ratio_perstep,
                            logprob_old.sum(-1),
                            logprob_new.sum(-1),
                        )

                        if chunk_adv_mode == 'per_step_vdelta':
                            loss = 0
                            for j in range(opt_steps):
                                step_adv = advantages[j].detach()
                                if step_adv.ndim > 1:
                                    step_adv = step_adv.squeeze(-1)
                                loss1 = ratio_perstep[:, j] * step_adv
                                loss2 = torch.clamp(
                                    ratio_perstep[:, j],
                                    1 - self._clip_ratio,
                                    1 + self._clip_ratio,
                                ) * step_adv
                                loss += -(torch.min(loss1, loss2)).mean()
                            loss = loss / opt_steps
                        else:
                            adv = advantages.detach().reshape(advantages.shape[0], -1)
                            assert adv.shape[1] == 1, \
                                f"chunk_as_single_action expects scalar advantage, got {adv.shape}"
                            adv = adv[:, 0].unsqueeze(-1)
                            loss1 = ratio_perstep * adv
                            loss2 = torch.clamp(ratio_perstep, 1 - self._clip_ratio, 1 + self._clip_ratio) * adv
                            loss = -(torch.min(loss1, loss2)).mean()
                elif opt_steps > 1:
                    # Per-step joint action ratio.  The scheduler returns
                    # per-event log-probs, so sum event dims before exp().
                    action_start = 0 if self.cfg.no_pre_action else self.n_obs_steps - 1
                    action_end = action_start + opt_steps
                    old_step_logprob = self._sum_step_logprob_event_dims(
                        old_all_logprob[i][:, action_start:action_end]
                    )
                    new_step_logprob = self._sum_step_logprob_event_dims(
                        new_log_prob[:, action_start:action_end]
                    )
                    ratio_perstep = (new_step_logprob - old_step_logprob).exp()
                    self._record_ratio_stats(
                        "offline_multi",
                        i,
                        ratio_perstep,
                        old_step_logprob,
                        new_step_logprob,
                    )
                    loss = 0
                    for j in range(opt_steps):
                        if self.cfg.chunk_as_single_action:
                            step_adv = advantages.detach()
                        else:
                            step_adv = advantages[j].detach()
                        step_adv = step_adv.reshape(step_adv.shape[0], -1)
                        assert step_adv.shape[1] == 1, \
                            f"offline PPO expects scalar step advantage, got {step_adv.shape}"
                        step_adv = step_adv[:, 0]
                        ratio_j = ratio_perstep[:, j]
                        loss1 = ratio_j * step_adv
                        loss2 = torch.clamp(ratio_j, 1 - self._clip_ratio, 1 + self._clip_ratio) * step_adv
                        loss += -(torch.min(loss1, loss2)).mean()
                    loss = loss / opt_steps
                else:
                    if self.cfg.no_pre_action:
                        action_idx = 0
                    else:
                        action_idx = self.n_obs_steps - 1
                    old_action_logprob = self._sum_logprob_event_dims(old_all_logprob[i][:, action_idx])
                    new_action_logprob = self._sum_logprob_event_dims(new_log_prob[:, action_idx])
                    action_ratio = (new_action_logprob - old_action_logprob).exp()
                    self._record_ratio_stats(
                        "offline_multi",
                        i,
                        action_ratio,
                        old_action_logprob,
                        new_action_logprob,
                    )
                    adv = advantages.detach().reshape(advantages.shape[0], -1)
                    assert adv.shape[1] == 1, \
                        f"offline PPO expects scalar advantage, got {adv.shape}"
                    adv = adv[:, 0]
                    loss1 = action_ratio * adv
                    loss2 = torch.clamp(action_ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * adv
                    loss = -(torch.min(loss1, loss2)).mean()
                if is_bc_loss:
                    loss = bc_loss + loss
                # pdb.set_trace()
                # torch.autograd.set_detect_anomaly(True)
                self._optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self._policy.model.parameters(), 0.5)
                self._optimizer.step()
                losses.append(loss.item())
        loss = sum(losses) / len(losses)

        if is_lr_decay:
            self._scheduler.step()
        if is_linear_decay:
            for p in self._optimizer.param_groups:
                p['lr'] = bppo_lr_now

        return loss if isinstance(loss, float) else loss.item()
    def save_critic(self, path):
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
        cprint('2ave critic to {}'.format(path), 'green')
    def load_critic(self, path):
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth')))
        cprint('load critic from {}'.format(path), 'green')
    def transfer2online(self, critic, dynamics, cfg, cm_optimizer=None, cm_lr_scheduler=None):
        # set policy, critic, dynamics to the new policy
        # self._policy = deepcopy(policy)
        self.cfg = cfg
        if cfg.distill_phase == 'online':
            self.distill = True
            self.cm_optimizer = cm_optimizer
            self.cm_lr_scheduler = cm_lr_scheduler
            self.update_phase = cfg.update_phase # step: update during each step, iteration, epoch, end: update at the end of mini-batch training
        else:
            self.distill = False
        self.kl_scheduler_actor = AdaptiveScheduler(kl_threshold=cfg.ppo.adaptive_kl, min_lr=cfg.ppo.min_lr_actor, max_lr=cfg.ppo.max_lr_actor,
                                              init_lr=cfg.ppo.lr_a)
        self.kl_scheduler_critic = AdaptiveScheduler(kl_threshold=cfg.ppo.adaptive_kl, min_lr=cfg.ppo.min_lr_critic, max_lr=cfg.ppo.max_lr_critic,
                                              init_lr=cfg.ppo.lr_a)
        self._policy.to(self._device)
        self._old_policy.to(self._device)
        self.critic = deepcopy(critic).to(self._device)
        self.dynamics = dynamics
        self.args = cfg.ppo
        self._policy.obs_encoder.eval() # eval encoder for online finetuning; 
        if self.args.encoder_lr_scale != 1.0:
            self.optimizer_encoder = torch.optim.Adam(
                self._policy.obs_encoder.parameters(),
                lr=self.args.lr_a * self.args.encoder_lr_scale, eps=1e-5)
            self.optimizer_actor = torch.optim.Adam(
                self._policy.model.parameters(),
                lr=self.args.lr_a, eps=1e-5) 
            cprint('encoder lr scale: {}'.format(self.args.encoder_lr_scale), 'yellow')
        else:        
            # import pdb; pdb.set_trace()
            if self.args.fix_encoder:
                self.optimizer_actor = torch.optim.Adam(
                    self._policy.model.parameters(),
                    lr=self.args.lr_a, eps=1e-5)
            else:
                # models_params = (
                # list(self._policy.obs_encoder.parameters()) +
                # list(self._policy.model.parameters())

                # )
                print('do not fix encoder for online finetuning')
                self.optimizer_actor = torch.optim.Adam(
                    self._policy.parameters(),
                    lr=self.args.lr_a, eps=1e-5)

        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_c, eps=1e-5)

        # ---- lighting-aug: optional frozen rgb backbone ----
        self.freeze_rgb_backbone = bool(getattr(cfg.ppo, 'freeze_rgb_backbone', False))
        if self.freeze_rgb_backbone and not self.args.fix_encoder:
            if self.args.encoder_lr_scale != 1.0:
                raise ValueError('ppo.freeze_rgb_backbone requires ppo.encoder_lr_scale == 1.0')
            enc = self._policy.obs_encoder
            if not hasattr(enc, 'freeze_backbone'):
                raise TypeError(f'{type(enc).__name__} has no freeze_backbone(); '
                                'freeze_rgb_backbone is only for MultiImageObsEncoder')
            n_frozen = enc.freeze_backbone()
            trainable = [p for p in self._policy.parameters() if p.requires_grad]
            self.optimizer_actor = torch.optim.Adam(trainable, lr=self.args.lr_a, eps=1e-5)
            cprint(f'[freeze_rgb_backbone] froze {n_frozen} backbone tensors; '
                   f'actor optimizer now has {len(trainable)} tensors', 'yellow')
        elif self.freeze_rgb_backbone:
            cprint('[freeze_rgb_backbone] ignored because ppo.fix_encoder=True already excludes the encoder', 'yellow')

        # ---- lighting-aug: optional prediction-consistency auxiliary ----
        aux_cfg = getattr(cfg.ppo, 'aux_consistency', None)
        if aux_cfg is None:
            self.aux_cfg = {}
        elif isinstance(aux_cfg, DictConfig):
            self.aux_cfg = OmegaConf.to_container(aux_cfg, resolve=True)
        else:
            self.aux_cfg = dict(aux_cfg)
        self.aux_enabled = bool(self.aux_cfg.get('enabled', False))
        self.aux_step = 0
        self.last_aux_metrics = {}
        self._aux_metric_buffer = defaultdict(list)
        self.aux_photometric = None
        self.aux_generator = None
        if self.aux_enabled:
            from rl_100.model.common.photometric_aug import PhotometricAug
            la_node = getattr(cfg.policy, 'lighting_aug', None)
            if la_node is None:
                raise ValueError('ppo.aux_consistency.enabled=True requires policy.lighting_aug in the config '
                                 '(only rl100_2d_*.yaml define it; aux_consistency is 2D-only)')
            la_cfg = OmegaConf.to_container(la_node, resolve=True) if isinstance(la_node, DictConfig) else dict(la_node)
            self.aux_photometric = PhotometricAug(la_cfg)          # reuses the offline range table; ignores la_cfg['enabled']
            profile = self.aux_cfg.get('lighting_profile', 'mild')
            if profile not in PhotometricAug.PROFILES:
                raise ValueError(f'ppo.aux_consistency.lighting_profile must be one of {PhotometricAug.PROFILES}, got {profile!r}')
            if not hasattr(self._policy, 'build_consistency_views'):
                raise TypeError(f'aux_consistency requires RL1002D, got {type(self._policy).__name__}')
            frac = float(self.aux_cfg.get('batch_fraction', 0.5))
            if not (0.0 < frac <= 1.0):
                raise ValueError(f'ppo.aux_consistency.batch_fraction must be in (0, 1], got {frac}')
            seed = int(getattr(cfg.training, 'seed', 0)) + 20260930
            self.aux_generator = torch.Generator(device=self._device)
            self.aux_generator.manual_seed(seed)
            cprint(f'[aux_consistency] enabled: {self.aux_cfg}', 'yellow')

    def warmup_critic(self, replay_buffer, gradient_steps):
        """Fit only the critic on the first full online rollout buffer."""
        gradient_steps = int(gradient_steps)
        if gradient_steps <= 0:
            return 0.0

        s, _, _, r, s_, dw, done = replay_buffer.numpy_to_tensor(
            device='cpu')
        r = r.to(self._device)
        dw = dw.to(self._device)
        done = done.to(self._device)
        with torch.no_grad():
            if self.args.share_encoder:
                vs, vs_ = self._compute_critic_values_in_chunks(
                    s, s_, use_obs2latent=True)
            else:
                vs, vs_ = self._compute_critic_values_in_chunks(
                    s, s_, use_obs2latent=False)

            gamma = self.args.gamma ** self.cfg.n_action_steps
            deltas = r + gamma * (1.0 - dw) * vs_ - vs
            advantages = []
            gae = 0
            for delta, d in zip(
                    reversed(deltas.flatten().cpu().numpy()),
                    reversed(done.flatten().cpu().numpy())):
                gae = delta + gamma * self.args.lamda * gae * (1.0 - d)
                advantages.insert(0, gae)
            advantages = torch.tensor(
                advantages, dtype=torch.float, device=self._device).view(-1, 1)
            value_targets = advantages + vs

        batch_size = r.shape[0]
        mini_batch_size = min(int(self.args.mini_batch_size), batch_size)
        critic_losses = []
        self.critic.train()
        for _ in tqdm(range(gradient_steps), desc='Critic warmup'):
            index = torch.randint(
                batch_size, (mini_batch_size,), device=self._device)
            index_cpu = index.cpu()
            state = dict_apply(
                s, lambda x: x[index_cpu].to(self._device))

            if self.args.share_encoder:
                with torch.no_grad():
                    critic_input = self._policy.obs2latent(state)
            else:
                critic_input = self._policy.obs2this_nobs(state)

            if self.args.value_recon:
                assert not self.args.share_encoder, \
                    "value_recon=True requires share_encoder=False so critic has its own obs_encoder."
                if isinstance(self.critic, torch.nn.Sequential):
                    critic_recon_loss, _, critic_features = \
                        self.critic[0].Recon_VIB_loss(critic_input)
                    value_pred = self.critic[1](critic_features)
                else:
                    critic_recon_loss, _, critic_features = \
                        self.critic._obs_encoder.Recon_VIB_loss(critic_input)
                    value_pred = self.critic(critic_features)
            else:
                value_pred = self.critic(critic_input)

            critic_loss = F.mse_loss(value_pred, value_targets[index].detach().float())
            if self.args.value_recon:
                critic_loss += critic_recon_loss

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            if self.args.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()
            critic_losses.append(critic_loss.item())

        if self.args.fix_encoder and self.args.v_encoder:
            self._policy.obs_encoder.load_state_dict(self.critic[0].state_dict())
        return float(np.mean(critic_losses))

    def dp_align_update_no_share(self, replay_buffer, total_steps, precomputed=None):
        self.iteration += 1
        # bc training before ppo improvement
        loss_metric = {}
        if self.args.use_bc:
            cprint('bc training', 'yellow')
            if not self.args.fix_encoder and self.args.encoder_lr_scale != 1.0 and self.args.actor_grad:
                loss_metric = self._policy.train_align(replay_buffer, self.optimizer_actor, self.args.fix_encoder, self.args.batch_size, encoder_optimizer=self.optimizer_encoder, iterations = self.args.iterations, mini_batch_size=self.args.mini_batch_size)
            else:
                loss_metric = self._policy.train_align(replay_buffer, self.optimizer_actor, self.args.fix_encoder, self.args.batch_size, iterations = self.args.iterations, mini_batch_size=self.args.mini_batch_size)
        else:
            loss_metric['bc_loss'] = 0
        # Keep high-resolution replay observations on CPU. Moving all current
        # and next images for the full rollout to CUDA creates a multi-GB
        # allocation before the first PPO mini-batch.
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor(
            device='cpu')
        a = a.to(self._device)
        a_logprob = a_logprob.to(self._device)
        r = r.to(self._device)
        dw = dw.to(self._device)
        done = done.to(self._device)
        a, a_logprob = a.transpose(0,1), a_logprob.transpose(0, 1)

        if precomputed is not None:
            # Vec env path: use precomputed per-env GAE
            adv = precomputed['adv']
            v_target = precomputed['v_target']
            vs = precomputed['vs']
            if self.args.use_adv_norm:
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        else:
            # Original single-env path
            adv = []
            gae = 0
            with torch.no_grad():
                if self.args.share_encoder:
                    vs, vs_ = self._compute_critic_values_in_chunks(s, s_, use_obs2latent=True)
                else:
                    vs, vs_ = self._compute_critic_values_in_chunks(s, s_, use_obs2latent=False)

                deltas = r + (self.args.gamma ** self.cfg.n_action_steps) * (1.0 - dw) * vs_ - vs
                for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                    gae = delta + (self.args.gamma ** self.cfg.n_action_steps) * self.args.lamda * gae * (1.0 - d)
                    adv.insert(0, gae)
                adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self._device)
                v_target = adv + vs
                if self.args.use_adv_norm:
                    adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        distill_loss = 0
        # Optimize policy    for K epochs:
        actor_losses, critic_losses = [], []
        ppo_start = time.time()
        for _ in tqdm(range(self.args.K_epochs), desc='PPO update'):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.args.batch_size)), self.args.mini_batch_size, False):
                actions = a[:, index, :]
                state = dict_apply(
                    s, lambda x: x[index].to(self._device))
                local_cond = None
                a_logprob_old = a_logprob[:, index, :]
                approx_kl_divs = []
                aux_ctx = self._prepare_aux_minibatch(state) if self.aux_enabled else None
                for i, t in enumerate(self._policy.noise_scheduler.timesteps):
                    timesteps = t
                    if not torch.is_tensor(timesteps):
                        timesteps = torch.tensor([timesteps], dtype=torch.long, device=self._device)
                    elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                        timesteps = timesteps[None].to(self._device)
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timesteps = timesteps.expand(actions.shape[1])
                    unet_timesteps = self._policy.get_unet_timesteps(timesteps)
                    if not self.args.fix_encoder and self.args.recon and (self.args.per_step_recon or i == 0):
                        obs_feature, recon_loss, recon_loss_item = self._policy.obs2latent_recon(state)
                    else:
                        obs_feature = self._policy.obs2latent(state)
                    model_output = self._policy.model(sample=actions[i],
                                timestep=unet_timesteps,
                                local_cond=local_cond, global_cond=obs_feature)
                    if hasattr(self._policy, 'diffusion_eta'):
                        if self._policy.diffusion_eta:
                            eta = torch.mean(model_output[:, :, -1], dim=1, keepdim=True)
                    else:   
                        eta = 1.
                    a_logprob_now, dist_entropy = self._policy.noise_scheduler.step_forward_logprob_with_entropy(model_output, 
                                                        timesteps, 
                                                        actions[i], 
                                                        next_sample=actions[i+1], eta=eta)
                                                        

                    action_start = 0 if self.cfg.no_pre_action else self.n_obs_steps - 1
                    action_end = action_start + self.cfg.n_action_steps

                    if self.cfg.chunk_as_single_action:
                        # chunk-as-single-action: use whole-chunk scalar ratio
                        logprob_now_chunk = a_logprob_now[:, action_start:action_end]
                        logprob_old_chunk = a_logprob_old[i][:, action_start:action_end]

                        logprob_now_scalar = logprob_now_chunk.reshape(logprob_now_chunk.shape[0], -1).sum(dim=1)  # [B]
                        logprob_old_scalar = logprob_old_chunk.reshape(logprob_old_chunk.shape[0], -1).sum(dim=1)  # [B]

                        ratios = torch.exp(logprob_now_scalar - logprob_old_scalar)  # [B]

                        self._record_ratio_stats(
                            "online",
                            i,
                            ratios,
                            logprob_old_scalar,
                            logprob_now_scalar,
                        )

                        adv_chunk = adv[index].reshape(-1)  # [B]
                        surr1 = ratios * adv_chunk
                        surr2 = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * adv_chunk
                    else:
                        # keep original non-chunk behavior unchanged
                        ratios = torch.exp(
                            a_logprob_now[:, action_start:action_end].sum(-1, keepdim=True)
                            - a_logprob_old[i][:, action_start:action_end].sum(-1, keepdim=True)
                        ).squeeze(1)

                        self._record_ratio_stats(
                            "online",
                            i,
                            ratios,
                            a_logprob_old[i][:, action_start:action_end].sum(-1),
                            a_logprob_now[:, action_start:action_end].sum(-1),
                        )

                        surr1 = ratios * adv[index]
                        surr2 = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * adv[index]

                    actor_loss = -torch.min(surr1, surr2) #- self.entropy_coef * dist_entropy  
                    if not self.args.fix_encoder and self.args.recon and (self.args.per_step_recon or i == 0):
                        actor_loss = actor_loss + recon_loss
                    actor_losses.append(actor_loss.mean().item())
                    # kl lr scheduler
                    if not self.args.use_lr_decay:
                        with torch.no_grad():
                            log_ratio = (
                                a_logprob_now[:, action_start:action_end].sum(-1, keepdim=True)
                                - a_logprob_old[i][:, action_start:action_end].sum(-1, keepdim=True)
                            ).squeeze(1)
                            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                            approx_kl_divs.append(approx_kl_div)
                        lr_actor = self.kl_scheduler_actor.update(approx_kl_div)
                        for param_group in self.optimizer_actor.param_groups:
                            param_group['lr'] = lr_actor
                        if self.args.encoder_lr_scale != 1.0:
                            for param_group in self.optimizer_encoder.param_groups:
                                param_group['lr'] = lr_actor * self.args.encoder_lr_scale

                    if aux_ctx is not None:
                        aux_loss, aux_info = self._aux_consistency_step(aux_ctx, actions[i], unet_timesteps, local_cond)
                        lam = self._aux_lambda()
                        grad_log_every = int(self.aux_cfg.get('grad_log_every', 0) or 0)
                        if grad_log_every > 0 and self.aux_step % grad_log_every == 0:
                            aux_info.update(self._aux_grad_norms(actor_loss.mean(), lam * aux_loss))
                        actor_loss = actor_loss + lam * aux_loss   # [B] + scalar: .mean() == ppo_mean + lam * aux_loss
                        aux_info['aux/lambda'] = lam
                        self._accumulate_aux_metrics(aux_info)

                    #Actor Gradient step
                    if not self.args.fix_encoder and self.args.encoder_lr_scale != 1.0:
                        self.optimizer_encoder.zero_grad()
                    self.optimizer_actor.zero_grad()
                    actor_loss.mean().backward()
                    nn.utils.clip_grad_norm_(list(self._policy.obs_encoder.parameters()) + list(self._policy.model.parameters()), 0.5)
                    self.optimizer_actor.step()
                    if aux_ctx is not None:
                        self.aux_step += 1

                    # encoder update if not fixed and encoder_lr_scale is not 1.0
                    if not self.args.fix_encoder:
                        if self.args.encoder_lr_scale != 1.0:
                            if self.args.actor_grad:
                                torch.nn.utils.clip_grad_norm_(self._policy.obs_encoder.parameters(), 0.5)
                                self.optimizer_encoder.step()
                    if self.distill:
                        if self.update_phase == 'step':
                            batch = {'obs':state, 'action':actions[-1]}
                            distill_loss = self.distill_update(batch=batch, online=True)
                if aux_ctx is not None and bool(self.aux_cfg.get('log_post_update_kl', False)):
                    self._accumulate_aux_metrics(
                        {'aux/post_update_approx_kl': self._post_update_approx_kl(state, actions, a_logprob_old)})
                if self.distill:
                    if self.update_phase == 'iteration':
                        batch = {'obs':state, 'action':actions[-1]}
                        distill_loss = self.distill_update(batch=batch, online=True)            
                if not self.args.use_lr_decay:
                    mean_appox_kl_div = np.mean(approx_kl_divs)
                    lr_critic = self.kl_scheduler_critic.update(mean_appox_kl_div)
                    for param_group in self.optimizer_critic.param_groups:
                        param_group['lr'] = lr_critic
                if not self.args.share_encoder:
                    critic_input = self._policy.obs2this_nobs(state) # pre-processed input
                else:
                    with torch.no_grad():
                        critic_input = self._policy.obs2latent(state)
                # do not share encoder; cirtic = {encoder, critic_mlp}
                if self.args.value_recon:
                    assert not self.args.share_encoder, \
                        "value_recon=True requires share_encoder=False so critic has its own obs_encoder."
                    if isinstance(self.critic, torch.nn.Sequential):
                        cricit_vib_recon_loss, critic_recon_loss_items, critic_nobs_features = self.critic[0].Recon_VIB_loss(critic_input)
                        v_s = self.critic[1](critic_nobs_features)
                    else:
                        cricit_vib_recon_loss, critic_recon_loss_items, critic_nobs_features = self.critic._obs_encoder.Recon_VIB_loss(critic_input)
                        v_s = self.critic(critic_nobs_features)
                else:
                    v_s = self.critic(critic_input)  # Get critic value

                
                if self.args.is_clip_value:
                    old_value_clipped = vs[index] + (v_s - vs[index]).clamp(-self.args.epsilon, self.args.epsilon)
                    value_loss = (v_s - v_target[index].detach().float()).pow(2)
                    value_loss_clipped = (old_value_clipped - v_target[index].detach().float()).pow(2)
                    critic_loss = torch.max(value_loss,value_loss_clipped).mean()
                else:
                    critic_loss = F.mse_loss(v_target[index], v_s)
                if self.args.value_recon:
                    critic_loss += cricit_vib_recon_loss
                # import pdb; pdb.set_trace() 
                critic_losses.append(critic_loss.mean().item())
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.args.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
        if self.args.fix_encoder:
            if self.args.v_encoder:
                self._policy.obs_encoder.load_state_dict(self.critic[0].state_dict())
        self.last_ppo_elapsed = time.time() - ppo_start
        if self.args.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)
        self.last_aux_metrics = self._finalize_aux_metrics()
        if self.distill:
            return np.mean(actor_losses), np.mean(critic_losses) , loss_metric['bc_loss'], distill_loss
        else:
            return np.mean(actor_losses), np.mean(critic_losses) , loss_metric['bc_loss'], 0

    # ------------------------------------------------------------------ lighting-aug auxiliary
    @contextmanager
    def _aux_rng_scope(self):
        """Keep the PPO main branch's global random stream untouched by the auxiliary (R6).

        The auxiliary draws its own randomness from ``self.aux_generator``, but the denoiser may
        contain dropout (MLPResNet) and VIB heads sample noise in training mode, both reading the
        *global* RNG. Forking the global RNG state around the auxiliary forward passes makes a run
        with the auxiliary consume exactly the same global random numbers as a run without it.
        """
        dev = torch.device(self._device)
        devices = []
        if dev.type == 'cuda':
            devices = [dev.index if dev.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            yield

    def _aux_draw_seed(self):
        return int(torch.randint(0, 2 ** 31 - 1, (1,), device=self._device, generator=self.aux_generator).item())

    def _seed_local_rng(self, seed):
        """Seed only the CPU generator and (if used) the auxiliary's own CUDA device (never *_all)."""
        torch.default_generator.manual_seed(seed)
        dev = torch.device(self._device)
        if dev.type == 'cuda':
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            torch.cuda.default_generators[idx].manual_seed(seed)

    def _aux_lambda(self):
        lam = float(self.aux_cfg.get('lambda_actor', 0.0))
        warm = int(self.aux_cfg.get('warmup_steps', 0) or 0)
        if warm > 0:
            lam = lam * min(1.0, (self.aux_step + 1) / warm)
        return lam

    def _prepare_aux_minibatch(self, state):
        """Pick the auxiliary subset of the minibatch, build (clean, aug) views and cache what the
        encoder configuration allows (see spec 5.4 for the four modes)."""
        B = state['agent_pos'].shape[0]
        frac = float(self.aux_cfg.get('batch_fraction', 0.5))
        n_aux = max(1, int(round(B * frac)))
        aux_idx = torch.randperm(B, device=self._device, generator=self.aux_generator)[:n_aux]
        state_aux = dict_apply(state, lambda x: x[aux_idx])
        shift_prob = float(self.aux_cfg.get('random_shift_prob', 0.5))
        shift_mask = torch.rand(n_aux, device=self._device, generator=self.aux_generator) < shift_prob
        profile = self.aux_cfg.get('lighting_profile', 'mild')
        views_c, views_a, stats = self._policy.build_consistency_views(
            state_aux, shift_mask, lighting_profile=profile,
            photometric=self.aux_photometric, generator=self.aux_generator)
        enc = self._policy.obs_encoder
        if hasattr(enc, 'has_trainable_params'):
            enc_has_trainable = enc.has_trainable_params()
        else:
            enc_has_trainable = any(p.requires_grad for p in enc.parameters())
        encoder_trainable = (not self.args.fix_encoder) and enc_has_trainable
        ctx = {'aux_idx': aux_idx, 'n_aux': n_aux, 'views_c': views_c, 'views_a': views_a,
               'n_shift': int(shift_mask.sum().item()), 'stats': stats}
        with self._policy._deterministic_obs_encoder():
            if not encoder_trainable:
                ctx['mode'] = 'cached_full'
                with torch.no_grad():
                    ctx['feat_c'] = self._policy.encode_flat(views_c).reshape(n_aux, -1)
                    ctx['feat_a'] = self._policy.encode_flat(views_a).reshape(n_aux, -1)
            elif self.freeze_rgb_backbone and hasattr(enc, 'backbone_forward'):
                ctx['mode'] = 'cached_backbone'
                with torch.no_grad():
                    ctx['bb_c'] = enc.backbone_forward(views_c)
                    ctx['bb_a'] = enc.backbone_forward(views_a)
            else:
                ctx['mode'] = 'full'
        return ctx

    def _aux_consistency_step(self, ctx, x_i, unet_timesteps, local_cond):
        """MSE between the denoiser output on the lighting-augmented view and the (stop-grad)
        output on the clean view, at the current denoise step. Does not touch any main-branch tensor."""
        idx, n = ctx['aux_idx'], ctx['n_aux']
        x = x_i[idx]
        t = unet_timesteps[idx] if torch.is_tensor(unet_timesteps) and unet_timesteps.ndim > 0 else unet_timesteps
        enc = self._policy.obs_encoder
        with self._aux_rng_scope():
            with self._policy._deterministic_obs_encoder():
                if ctx['mode'] == 'cached_full':
                    feat_c, feat_a = ctx['feat_c'], ctx['feat_a']
                elif ctx['mode'] == 'cached_backbone':
                    with torch.no_grad():
                        feat_c = enc.head_forward(ctx['bb_c'], ctx['views_c']).reshape(n, -1)
                    feat_a = enc.head_forward(ctx['bb_a'], ctx['views_a']).reshape(n, -1)
                else:
                    with torch.no_grad():
                        feat_c = self._policy.encode_flat(ctx['views_c']).reshape(n, -1)
                    feat_a = self._policy.encode_flat(ctx['views_a']).reshape(n, -1)
            # identical dropout masks (if any) for target and prediction
            seed = self._aux_draw_seed()
            self._seed_local_rng(seed)
            with torch.no_grad():
                target = self._policy.model(sample=x, timestep=t, local_cond=local_cond, global_cond=feat_c)
            self._seed_local_rng(seed)
            pred = self._policy.model(sample=x, timestep=t, local_cond=local_cond, global_cond=feat_a)
        aux_loss = F.mse_loss(pred, target)
        info = {
            'aux/loss': aux_loss.item(),
            'aux/feat_l2': (feat_a.detach() - feat_c).norm(dim=-1).mean().item(),
            'aux/pred_l2': (pred.detach() - target).flatten(1).norm(dim=-1).mean().item(),
            'aux/n_shift': ctx['n_shift'], 'aux/n_aux': n,
        }
        info.update({f'aux/{k}': float(v) for k, v in ctx['stats'].items()})
        info['aux/mode_' + ctx['mode']] = 1.0
        return aux_loss, info

    def _aux_grad_norms(self, ppo_loss_mean, scaled_aux_loss):
        params = [p for p in self._policy.parameters() if p.requires_grad]

        def _norm(grads):
            sq = [g.pow(2).sum() for g in grads if g is not None]
            return float(torch.sqrt(torch.stack(sq).sum()).item()) if sq else 0.0

        g_ppo = torch.autograd.grad(ppo_loss_mean, params, retain_graph=True, allow_unused=True)
        g_aux = torch.autograd.grad(scaled_aux_loss, params, retain_graph=True, allow_unused=True)
        return {'aux/ppo_grad_norm': _norm(g_ppo), 'aux/aux_grad_norm': _norm(g_aux)}

    def _post_update_approx_kl(self, state, actions, a_logprob_old):
        """Diagnostic: approx KL between the just-updated policy and the rollout log-probs on this
        minibatch, averaged over denoise steps. Flow schedulers only (F17); otherwise nan."""
        if not getattr(self._policy, 'is_flow', False):
            if not getattr(self, '_warned_post_update_kl', False):
                cprint('[aux_consistency] log_post_update_kl only supports flow schedulers; logging nan', 'yellow')
                self._warned_post_update_kl = True
            return float('nan')
        action_start = 0 if self.cfg.no_pre_action else self.n_obs_steps - 1
        action_end = action_start + self.cfg.n_action_steps
        kls = []
        with self._aux_rng_scope(), torch.no_grad():
            for i, t in enumerate(self._policy.noise_scheduler.timesteps):
                timesteps = t
                if not torch.is_tensor(timesteps):
                    timesteps = torch.tensor([timesteps], dtype=torch.long, device=self._device)
                elif timesteps.ndim == 0:
                    timesteps = timesteps[None].to(self._device)
                timesteps = timesteps.expand(actions.shape[1])
                unet_timesteps = self._policy.get_unet_timesteps(timesteps)
                obs_feature = self._policy.obs2latent(state)
                model_output = self._policy.model(sample=actions[i], timestep=unet_timesteps,
                                                  local_cond=None, global_cond=obs_feature)
                eta = 1.
                if getattr(self._policy, 'diffusion_eta', False):
                    eta = torch.mean(model_output[:, :, -1], dim=1, keepdim=True)
                a_logprob_now, _ = self._policy.noise_scheduler.step_forward_logprob_with_entropy(
                    model_output, timesteps, actions[i], next_sample=actions[i + 1], eta=eta)
                log_ratio = a_logprob_now[:, action_start:action_end] - a_logprob_old[i][:, action_start:action_end]
                log_ratio = log_ratio.reshape(log_ratio.shape[0], -1).sum(dim=1)
                kls.append(torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item())
        return float(np.mean(kls)) if kls else float('nan')

    def _accumulate_aux_metrics(self, info):
        for k, v in info.items():
            self._aux_metric_buffer[k].append(float(v))

    def _finalize_aux_metrics(self):
        out = {k: float(np.mean(v)) for k, v in self._aux_metric_buffer.items() if len(v)}
        self._aux_metric_buffer.clear()
        return out

    def distill_update(self, batch = None, online = False):
        if getattr(self._policy, 'is_flow', False):
            loss, loss_dict = self._policy.compute_flow_distill_loss(
                batch, distill2mean=self.cfg.distill2mean)
        elif self.cfg.distill_loss_type == 'back_up':
            loss, loss_dict = self._policy.compute_ddim2cm_loss(batch, online = online, distill2mean=self.cfg.distill2mean)
        elif self.cfg.distill_loss_type == 'action':
            loss, loss_dict = self._policy.compute_ddim2cm_loss_action(batch, online = online, distill2mean=self.cfg.distill2mean)
        elif self.cfg.distill_loss_type == 'action_same_noise':
            loss, loss_dict = self._policy.compute_ddim2cm_loss_action_same_noise(batch, online = online, distill2mean=self.cfg.distill2mean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.distilled_model.parameters(), self.cfg.training.max_grad_norm)
        self.cm_optimizer.step()
        self.cm_lr_scheduler.step()
        self.cm_optimizer.zero_grad(set_to_none=True)
        if not getattr(self._policy, 'is_flow', False):
            update_ema(self._policy.target_model.parameters(), self._policy.distilled_model.parameters(), self.cfg.training.ema_decay)
        return loss.item()
            
    def lr_decay(self, total_steps):
        lr_a_now = self.args.lr_a * (1 - total_steps / self.args.max_train_steps)
        lr_c_now = self.args.lr_c * (1 - total_steps / self.args.max_train_steps)
        self.lr_a = lr_a_now
        self.lr_c = lr_c_now
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
        if self.args.encoder_lr_scale != 1.0:
            for p in self.optimizer_encoder.param_groups:
                p['lr'] = lr_a_now * self.args.encoder_lr_scale
