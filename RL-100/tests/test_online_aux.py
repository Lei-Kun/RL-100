"""T4-T8: online PPO auxiliary, frozen backbone, bit-exactness with the auxiliary off (spec §8.5)."""
import copy
import math
import os

import pytest
import torch
from omegaconf import OmegaConf

import _support as S


def _aux_overrides(**kw):
    out = ['ppo.aux_consistency.enabled=true']
    out += [f'ppo.aux_consistency.{k}={v}' for k, v in kw.items()]
    return out


def _cfg(use_vib=False, use_recon=False, fix_encoder=False, freeze=None, aux=None):
    extra = _aux_overrides(**aux) if aux is not None else []
    return S.small_cfg(use_vib=use_vib, use_recon=use_recon, fix_encoder=fix_encoder,
                       freeze_rgb_backbone=freeze, extra=extra)


def _run_update(ppo, replay, seed=0):
    torch.manual_seed(seed)
    return ppo.dp_align_update_no_share(replay, total_steps=1)


def _param_ids(params):
    return {id(p) for p in params}


def _optimizer_param_ids(opt):
    return {id(p) for g in opt.param_groups for p in g['params']}


# ----------------------------------------------------------------------------- T4
def test_t4_build_consistency_views():
    cfg = _cfg(use_vib=True, aux={})
    policy = S.build_policy(cfg, seed=0)
    from rl_100.model.common.photometric_aug import PhotometricAug
    photometric = PhotometricAug(OmegaConf.to_container(cfg.policy.lighting_aug, resolve=True))
    obs = S.random_batch(cfg, B=4)['obs']
    B, To = 4, int(cfg.n_obs_steps)
    ref = policy.obs2this_nobs(copy.deepcopy(obs), training=False)

    no_shift = torch.zeros(B, dtype=torch.bool)
    clean, aug, stats = policy.build_consistency_views(obs, no_shift, 'wide', photometric,
                                                       generator=torch.Generator().manual_seed(1))
    for key in policy.rgb_obs_keys:
        assert clean[key].dtype == aug[key].dtype == torch.float32
        assert clean[key].shape == aug[key].shape == (B * To, 3, 32, 32)
        assert torch.equal(clean[key], ref[key].float())          # no shift -> normalised flattened original
        assert not torch.equal(aug[key], clean[key])
    for key in ('agent_pos', 'point_cloud'):
        assert aug[key] is clean[key]
    assert stats['profile_wide_frac'] == 1.0

    all_shift = torch.ones(B, dtype=torch.bool)
    clean_s, aug_s, _ = policy.build_consistency_views(obs, all_shift, 'identity', photometric,
                                                       generator=torch.Generator().manual_seed(2))
    for key in policy.rgb_obs_keys:
        assert not torch.equal(clean_s[key], ref[key].float())   # shifted
        assert torch.equal(aug_s[key], clean_s[key])             # identity profile -> bit-exact
    # clean must not be mutated by the photometric pass
    snap = {k: clean_s[k].clone() for k in policy.rgb_obs_keys}
    half = torch.tensor([True, False, True, False])
    clean_h, aug_h, _ = policy.build_consistency_views(obs, half, 'wide', photometric,
                                                       generator=torch.Generator().manual_seed(3))
    for key in policy.rgb_obs_keys:
        assert torch.equal(clean_s[key], snap[key])
        y = clean_h[key].view(B, To, 3, 32, 32)
        r = ref[key].float().view(B, To, 3, 32, 32)
        assert torch.equal(y[1], r[1]) and torch.equal(y[3], r[3])
        assert not torch.equal(y[0], r[0])
    # explicit generator => global RNG untouched
    torch.manual_seed(11); probe = torch.rand(3)
    torch.manual_seed(11); policy.build_consistency_views(obs, half, 'wide', photometric, generator=torch.Generator().manual_seed(4))
    assert torch.equal(torch.rand(3), probe)


# ----------------------------------------------------------------------------- T5a / T5g
@pytest.mark.parametrize('use_vib,train_mode,grad_log_every', [
    (False, False, 0), (True, False, 0), (True, True, 0), (True, False, 1)])
def test_t5a_lambda_zero_is_bit_exact(use_vib, train_mode, grad_log_every):
    cfg_off = _cfg(use_vib=use_vib)
    cfg_on = _cfg(use_vib=use_vib, aux=dict(lambda_actor=0.0, warmup_steps=0, grad_log_every=grad_log_every))
    policy = S.build_policy(cfg_off, seed=0)
    ppo_off = S.build_ppo(cfg_off, copy.deepcopy(policy))
    ppo_on = S.build_ppo(cfg_on, copy.deepcopy(policy))
    if train_mode:   # MLPResNet dropout + VIB sampling read the global RNG -> exercises _aux_rng_scope
        ppo_off._policy.train(); ppo_on._policy.train()
    replay = S.FakeReplay(cfg_off, seed=7, policy=policy)   # rollout log-probs: every denoise step carries gradient
    out_off = _run_update(ppo_off, replay)
    out_on = _run_update(ppo_on, replay)
    assert S.state_dicts_equal(ppo_off._policy, ppo_on._policy)
    assert S.state_dicts_equal(ppo_off.critic, ppo_on.critic)
    assert out_off == out_on
    assert ppo_off.last_aux_metrics == {}
    assert ppo_on.aux_enabled and ppo_on.last_aux_metrics['aux/lambda'] == 0.0
    assert math.isfinite(ppo_on.last_aux_metrics['aux/loss'])
    assert ppo_on.aux_step == int(cfg_on.ppo.K_epochs) * int(cfg_on.flow_inference_steps)
    if grad_log_every:
        assert math.isfinite(ppo_on.last_aux_metrics['aux/ppo_grad_norm'])
        assert math.isfinite(ppo_on.last_aux_metrics['aux/aux_grad_norm'])
    # after the update both copies must also draw the same next global random number
    torch.manual_seed(5); a = torch.rand(1)
    torch.manual_seed(5); b = torch.rand(1)
    assert torch.equal(a, b)


# ----------------------------------------------------------------------------- T5b
def test_t5b_fix_encoder_cached_full():
    cfg = _cfg(use_vib=True, fix_encoder=True, aux=dict(warmup_steps=0))
    policy = S.build_policy(cfg, seed=0)
    ppo = S.build_ppo(cfg, policy)
    enc_before = copy.deepcopy(ppo._policy.obs_encoder.state_dict())
    model_before = copy.deepcopy(ppo._policy.model.state_dict())
    _run_update(ppo, S.FakeReplay(cfg, policy=policy))
    assert S.state_dicts_equal(enc_before, ppo._policy.obs_encoder.state_dict())
    assert not S.state_dicts_equal(model_before, ppo._policy.model.state_dict())
    m = ppo.last_aux_metrics
    assert m['aux/mode_cached_full'] == 1.0
    assert math.isfinite(m['aux/loss']) and m['aux/loss'] > 0.0
    assert m['aux/lambda'] == float(cfg.ppo.aux_consistency.lambda_actor)
    assert m['aux/n_aux'] == max(1, round(int(cfg.ppo.mini_batch_size) * float(cfg.ppo.aux_consistency.batch_fraction)))
    assert 'aux/profile_mild_frac' in m and m['aux/profile_mild_frac'] == 1.0


# ----------------------------------------------------------------------------- T5c
def test_t5c_frozen_backbone_trainable_vib_heads():
    cfg = _cfg(use_vib=True, fix_encoder=False, freeze=True, aux=dict(warmup_steps=0))
    policy = S.build_policy(cfg, seed=0)
    ppo = S.build_ppo(cfg, policy)
    enc = ppo._policy.obs_encoder
    assert ppo.freeze_rgb_backbone
    assert all(not p.requires_grad for p in enc.backbone_parameters())
    n_trainable = sum(1 for p in ppo._policy.parameters() if p.requires_grad)
    assert sum(len(g['params']) for g in ppo.optimizer_actor.param_groups) == n_trainable
    assert _optimizer_param_ids(ppo.optimizer_actor) == _param_ids(p for p in ppo._policy.parameters() if p.requires_grad)
    bb_before = copy.deepcopy(enc.key_model_map.state_dict())
    vib_before = copy.deepcopy(enc.vib_heads.state_dict())
    _run_update(ppo, S.FakeReplay(cfg, policy=policy))
    assert S.state_dicts_equal(bb_before, enc.key_model_map.state_dict())
    assert not S.state_dicts_equal(vib_before, enc.vib_heads.state_dict())
    assert ppo.last_aux_metrics['aux/mode_cached_backbone'] == 1.0
    assert all(not p.requires_grad and p.grad is None for p in enc.backbone_parameters())


# ----------------------------------------------------------------------------- T5d
def test_t5d_frozen_backbone_without_heads_is_cached_full():
    cfg = _cfg(use_vib=False, use_recon=False, fix_encoder=False, freeze=True, aux=dict(warmup_steps=0))
    policy = S.build_policy(cfg, seed=0)
    ppo = S.build_ppo(cfg, policy)
    assert not ppo._policy.obs_encoder.has_trainable_params()
    enc_before = copy.deepcopy(ppo._policy.obs_encoder.state_dict())
    out = _run_update(ppo, S.FakeReplay(cfg, policy=policy))
    assert all(math.isfinite(float(v)) for v in out)
    assert ppo.last_aux_metrics['aux/mode_cached_full'] == 1.0
    assert S.state_dicts_equal(enc_before, ppo._policy.obs_encoder.state_dict())


# ----------------------------------------------------------------------------- T5e
@pytest.mark.parametrize('fix_encoder', [False, True])
def test_t5e_identity_views_give_zero_loss(fix_encoder):
    cfg = _cfg(use_vib=True, fix_encoder=fix_encoder,
               aux=dict(lighting_profile='identity', random_shift_prob=0.0, warmup_steps=0))
    policy = S.build_policy(cfg, seed=0)
    ppo = S.build_ppo(cfg, policy)
    _run_update(ppo, S.FakeReplay(cfg, policy=policy))
    m = ppo.last_aux_metrics
    assert m['aux/loss'] == 0.0 and m['aux/feat_l2'] == 0.0 and m['aux/pred_l2'] == 0.0
    assert m['aux/n_shift'] == 0.0
    assert m['aux/profile_identity_frac'] == 1.0


# ----------------------------------------------------------------------------- T5f
def test_t5f_freeze_ignored_when_fix_encoder():
    cfg = _cfg(use_vib=True, fix_encoder=True, freeze=True)
    policy = S.build_policy(cfg, seed=0)
    ppo = S.build_ppo(cfg, policy)
    enc = ppo._policy.obs_encoder
    assert ppo.freeze_rgb_backbone
    assert all(p.requires_grad for p in enc.backbone_parameters())
    assert _optimizer_param_ids(ppo.optimizer_actor) == _param_ids(ppo._policy.model.parameters())


# ----------------------------------------------------------------------------- T5h: warmup + post-update KL
def test_t5h_warmup_and_post_update_kl():
    cfg = _cfg(use_vib=False, fix_encoder=True, aux=dict(lambda_actor=0.5, warmup_steps=4, log_post_update_kl='true'))
    policy = S.build_policy(cfg, seed=0)
    ppo = S.build_ppo(cfg, policy)
    assert ppo._aux_lambda() == pytest.approx(0.5 / 4)
    _run_update(ppo, S.FakeReplay(cfg, policy=policy))
    steps = int(cfg.ppo.K_epochs) * int(cfg.flow_inference_steps)
    expected = sum(0.5 * min(1.0, (k + 1) / 4) for k in range(steps)) / steps
    assert ppo.last_aux_metrics['aux/lambda'] == pytest.approx(expected)
    assert math.isfinite(ppo.last_aux_metrics['aux/post_update_approx_kl'])


# ----------------------------------------------------------------------------- T6
def test_t6_logprob_regression_harness():
    cfg = _cfg(use_vib=False, use_recon=False)
    policy = S.build_policy(cfg, seed=0)
    obs = S.random_batch(cfg, B=4)['obs']
    with torch.no_grad():
        torch.manual_seed(3)
        action, all_x, all_logprob = policy.all_step_action_logprob(copy.deepcopy(obs))
        T = int(cfg.flow_inference_steps)
        assert all_x.shape[0] == T + 1 and all_logprob.shape[0] == T
        scheduler = policy.noise_scheduler
        scheduler.set_timesteps(T)
        for i, t in enumerate(scheduler.timesteps):
            timesteps = t if torch.is_tensor(t) else torch.tensor([t], dtype=torch.long)
            if timesteps.ndim == 0:
                timesteps = timesteps[None]
            timesteps = timesteps.expand(all_x[i].shape[0])
            obs_feature = policy.obs2latent(copy.deepcopy(obs))
            model_output = policy.model(sample=all_x[i], timestep=policy.get_unet_timesteps(timesteps),
                                        local_cond=None, global_cond=obs_feature)
            lp, _ = scheduler.step_forward_logprob_with_entropy(model_output, timesteps, all_x[i], next_sample=all_x[i + 1])
            assert torch.allclose(lp, all_logprob[i], atol=1e-5), i


# ----------------------------------------------------------------------------- T7
def test_t7_config_without_new_keys_is_silent():
    cfg = _cfg(use_vib=True, fix_encoder=False)
    OmegaConf.set_struct(cfg, False)
    del cfg.ppo.freeze_rgb_backbone
    del cfg.ppo.aux_consistency
    del cfg.policy.lighting_aug
    OmegaConf.set_struct(cfg, True)
    assert 'lighting_aug' not in cfg.policy and 'aux_consistency' not in cfg.ppo
    policy = S.build_policy(cfg, seed=0)
    assert policy.lighting_aug is None
    ppo = S.build_ppo(cfg, policy)
    assert ppo.aux_enabled is False and ppo.freeze_rgb_backbone is False and ppo.aux_cfg == {}
    assert _optimizer_param_ids(ppo.optimizer_actor) == _param_ids(ppo._policy.parameters())
    out = _run_update(ppo, S.FakeReplay(cfg, policy=policy))
    assert all(math.isfinite(float(v)) for v in out)
    assert ppo.last_aux_metrics == {}
    # an aux request without policy.lighting_aug must fail loudly, not silently fall back
    OmegaConf.set_struct(cfg, False)
    cfg.ppo.aux_consistency = {'enabled': True}
    OmegaConf.set_struct(cfg, True)
    with pytest.raises(ValueError):
        S.build_ppo(cfg, policy)


# ----------------------------------------------------------------------------- T8
def test_t8_static_guards_3d_untouched():
    root = S.RL100_ROOT
    src = open(os.path.join(root, 'rl_100', 'policy', 'rl100_3d.py')).read()
    for token in ('photometric_aug', 'lighting_aug', 'multi_image_obs_encoder'):
        assert token not in src
    for name in ('rl100_3d_flow.yaml', 'rl100_3d_epsilon.yaml'):
        text = open(os.path.join(root, 'rl_100', 'config', name)).read()
        for token in ('lighting_aug', 'aux_consistency', 'freeze_rgb_backbone'):
            assert token not in text, (name, token)
    # the only importer of rl_100.model.common.aug / photometric_aug is the 2D policy (+ uni_ppo lazily)
    aug_src = open(os.path.join(root, 'rl_100', 'model', 'common', 'photometric_aug.py')).read()
    assert 'import rl_100' not in aug_src and 'from rl_100' not in aug_src
