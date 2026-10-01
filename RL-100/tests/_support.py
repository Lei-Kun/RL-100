"""Shared helpers for the lighting-aug test-suite and tests/manual/dump_reference_hashes.py.

CPU only. Every shape / horizon / key is derived from the composed Hydra config so the
helpers work for any 2D task config; the defaults below are a tiny synthetic setting
(32x32 rgb, 64-point clouds, 3 flow steps) chosen for speed, not for realism.
"""
import hashlib
import os
import sys
import types

RL100_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CONFIG_DIR = os.path.join(RL100_ROOT, 'rl_100', 'config')
DEFAULT_CONFIG_NAME = os.environ.get('LIGHTING_AUG_TEST_CONFIG', 'rl100_2d_flow')
NUM_THREADS = int(os.environ.get('LIGHTING_AUG_TEST_THREADS', '8'))


def install_import_stubs():
    """Import-time shims for third-party packages that some rl_100 modules import at
    module level (``gym`` in rl_100/unidpg/utils.py, ``stable_baselines3`` in
    rl_100/model/vision/torch_layers.py) but that are absent on CPU-only test hosts.
    Real packages win whenever they are installed; the stubs only provide the names
    that are touched at import time and are never exercised by the tests.
    """
    if RL100_ROOT not in sys.path:
        sys.path.insert(0, RL100_ROOT)
    try:
        import gym  # noqa: F401
    except ModuleNotFoundError:
        try:
            import gymnasium as gym_compat
        except ModuleNotFoundError:
            gym_compat = types.ModuleType('gym')
        sys.modules['gym'] = gym_compat
    try:
        import stable_baselines3  # noqa: F401
    except ModuleNotFoundError:
        pkg = types.ModuleType('stable_baselines3'); pkg.__path__ = []
        common = types.ModuleType('stable_baselines3.common'); common.__path__ = []
        preprocessing = types.ModuleType('stable_baselines3.common.preprocessing')
        preprocessing.get_flattened_obs_dim = lambda *a, **k: 0
        preprocessing.is_image_space = lambda *a, **k: False
        type_aliases = types.ModuleType('stable_baselines3.common.type_aliases')
        type_aliases.TensorDict = dict
        utils = types.ModuleType('stable_baselines3.common.utils')
        utils.get_device = lambda *a, **k: 'cpu'
        pkg.common = common
        common.preprocessing, common.type_aliases, common.utils = preprocessing, type_aliases, utils
        for mod in (pkg, common, preprocessing, type_aliases, utils):
            sys.modules[mod.__name__] = mod


install_import_stubs()

import hydra  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)
torch.set_num_threads(NUM_THREADS)

# Tiny synthetic setting (spec §8.1). use_vib / use_recon / fix_encoder / freeze are arguments.
BASE_OVERRIDES = (
    'task=peg_2d',
    'training.device=cpu',
    'horizon=3', 'n_obs_steps=3', 'n_action_steps=1',
    'policy.model=skipnet', 'policy.scheduler_type=flow',
    'policy.img_shape=[3,32,32]',
    'task.shape_meta.obs.external_img.shape=[3,32,32]',
    'task.shape_meta.obs.wrist_img.shape=[3,32,32]',
    'task.shape_meta.obs.point_cloud.shape=[64,3]',
    'encoders.resnet.resize_shape=null',
    'encoders.resnet.rgb_model.weights=null',      # never download R3M in tests
    'flow_inference_steps=3', 'num_inference_steps=3',
    'ppo.batch_size=4', 'ppo.mini_batch_size=4', 'ppo.K_epochs=1', 'ppo.use_lr_decay=true',
)


def _yaml_bool(v):
    return str(bool(v)).lower()


def small_overrides(use_vib=False, use_recon=False, fix_encoder=None, freeze_rgb_backbone=None, extra=()):
    ov = list(BASE_OVERRIDES) + [f'use_vib={_yaml_bool(use_vib)}', f'use_recon={_yaml_bool(use_recon)}']
    if fix_encoder is not None:
        ov.append(f'ppo.fix_encoder={_yaml_bool(fix_encoder)}')
    if freeze_rgb_backbone is not None:
        ov.append(f'ppo.freeze_rgb_backbone={_yaml_bool(freeze_rgb_backbone)}')
    ov.extend(extra)
    return ov


def compose_cfg(overrides, config_name=DEFAULT_CONFIG_NAME):
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return hydra.compose(config_name=config_name, overrides=list(overrides))


def small_cfg(use_vib=False, use_recon=False, fix_encoder=None, freeze_rgb_backbone=None, extra=(),
              config_name=DEFAULT_CONFIG_NAME):
    return compose_cfg(small_overrides(use_vib, use_recon, fix_encoder, freeze_rgb_backbone, extra), config_name)


def shape_meta(cfg):
    return OmegaConf.to_container(cfg.shape_meta, resolve=True)


def fit_normalizer(cfg, n=32, seed=0):
    """LinearNormalizer over random action / non-rgb obs (images stay un-normalised, F9)."""
    from rl_100.model.common.normalizer import LinearNormalizer
    g = torch.Generator().manual_seed(seed)
    meta = shape_meta(cfg)
    data = {'action': torch.rand(n, *meta['action']['shape'], generator=g)}
    for key, attr in meta['obs'].items():
        if attr.get('type', 'low_dim') != 'rgb':
            data[key] = torch.rand(n, *attr['shape'], generator=g)
    norm = LinearNormalizer()
    norm.fit(data, last_n_dims=1, mode='limits')
    return norm


def build_policy(cfg, seed=0):
    torch.manual_seed(seed)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(fit_normalizer(cfg, seed=seed))
    policy.eval()
    return policy


def random_batch(cfg, B=4, seed=1, n_frames=None):
    """obs: (B, n_frames, *shape) in [0,1) for every shape_meta key; action: (B, horizon, Da).
    n_frames defaults to n_obs_steps (obs_as_global_cond=True layout)."""
    g = torch.Generator().manual_seed(seed)
    meta = shape_meta(cfg)
    frames = int(cfg.n_obs_steps) if n_frames is None else int(n_frames)
    obs = {key: torch.rand(B, frames, *attr['shape'], generator=g) for key, attr in meta['obs'].items()}
    action = torch.rand(B, int(cfg.horizon), *meta['action']['shape'], generator=g)
    return {'obs': obs, 'action': action}


class TinyCritic(nn.Module):
    """Minimal value head over the flattened this_nobs dict: agent_pos (B*To, D) -> (B, 1)."""

    def __init__(self, n_obs_steps, agent_pos_dim):
        super().__init__()
        self.n_obs_steps = int(n_obs_steps)
        self.lin = nn.Linear(self.n_obs_steps * int(agent_pos_dim), 1)

    def forward(self, obs):
        x = obs['agent_pos']
        return self.lin(x.reshape(-1, self.n_obs_steps * x.shape[-1]))


def build_ppo(cfg, policy, critic=None):
    """Construct BehaviorProximalPolicyOptimization the way train_real.py does and move it online."""
    from rl_100.unidpg.uni_ppo import BehaviorProximalPolicyOptimization
    ppo = BehaviorProximalPolicyOptimization(
        policy=policy,
        device=torch.device(cfg.training.device),
        policy_lr=cfg.unio4.bppo_lr,
        clip_ratio=cfg.unio4.clip_ratio,
        entropy_weight=cfg.unio4.entropy_weight,
        decay=cfg.unio4.decay,
        omega=cfg.unio4.omega,
        batch_size=cfg.unio4.bppo_batch_size,
        is_iql=cfg.critic.is_iql,
        temperature=cfg.unio4.temperature,
        ratio_strategy=cfg.unio4.ratio_strategy,
        top_k=cfg.unio4.top_k,
        num_inference_steps=cfg.policy.num_inference_steps,
        fix_encoder=cfg.unio4.fix_encoder,
        cfg=cfg,
    )
    ppo.set_old_policy()
    if critic is None:
        meta = shape_meta(cfg)
        critic = TinyCritic(cfg.n_obs_steps, meta['obs']['agent_pos']['shape'][0])
    ppo.transfer2online(critic, None, cfg)
    # mirror train_real.online_ft: VIB sampling stays stochastic in the online main branch
    enc = ppo._policy.obs_encoder
    if hasattr(enc, 'force_stochastic'):
        enc.force_stochastic = bool(getattr(cfg.ppo, 'force_stochastic_online', True))
    # the rollout normally calls set_timesteps; dp_align_update_no_share relies on it
    steps = cfg.flow_inference_steps if getattr(ppo._policy, 'is_flow', False) else cfg.num_inference_steps
    ppo._policy.noise_scheduler.set_timesteps(int(steps))
    return ppo


class FakeReplay:
    """Stand-in for the online ReplayBuffer: numpy_to_tensor() -> (s, a, a_logprob, r, s_, dw, done)
    with a (B, T+1, H, Da) and a_logprob (B, T, H, Da), T = flow_inference_steps, H = horizon - n_obs_steps + 1."""

    def __init__(self, cfg, B=None, seed=7):
        B = int(cfg.ppo.batch_size) if B is None else int(B)
        g = torch.Generator().manual_seed(seed)
        meta = shape_meta(cfg)
        Da = int(meta['action']['shape'][0])
        T = int(cfg.flow_inference_steps)
        H = int(cfg.horizon) - int(cfg.n_obs_steps) + 1
        self.s = random_batch(cfg, B, seed=seed)['obs']
        self.s_ = random_batch(cfg, B, seed=seed + 1)['obs']
        self.a = torch.randn(B, T + 1, H, Da, generator=g)
        self.a_logprob = torch.randn(B, T, H, Da, generator=g)
        self.r = torch.rand(B, 1, generator=g)
        self.dw = torch.zeros(B, 1)
        self.done = torch.zeros(B, 1)
        self.done[-1] = 1.0

    def numpy_to_tensor(self):
        return self.s, self.a, self.a_logprob, self.r, self.s_, self.dw, self.done


def state_dict_sha256(module_or_sd):
    sd = module_or_sd.state_dict() if hasattr(module_or_sd, 'state_dict') else module_or_sd
    h = hashlib.sha256()
    for k, v in sd.items():
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(str(v.dtype).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def state_dicts_equal(a, b):
    """True iff same keys and every tensor is torch.equal."""
    sa = a.state_dict() if hasattr(a, 'state_dict') else a
    sb = b.state_dict() if hasattr(b, 'state_dict') else b
    if list(sa.keys()) != list(sb.keys()):
        return False
    return all(torch.equal(sa[k], sb[k]) for k in sa)
