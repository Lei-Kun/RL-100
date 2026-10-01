"""T2: offline SFT gating of the lighting augmentation inside RL1002D.compute_loss (spec §8.3)."""
import copy

import torch

import _support as S

VIB_RECON = dict(use_vib=True, use_recon=True)   # production-like encoder path (Recon_VIB_loss)
ALL_WIDE = ('policy.lighting_aug.profile_prob.identity=0.0', 'policy.lighting_aug.profile_prob.mild=0.0')


def _pair(extra_on=()):
    cfg_off = S.small_cfg(**VIB_RECON)
    cfg_on = S.small_cfg(**VIB_RECON, extra=['policy.lighting_aug.enabled=true', *extra_on])
    p_off, p_on = S.build_policy(cfg_off, seed=0), S.build_policy(cfg_on, seed=0)
    assert p_off.lighting_aug is None and p_on.lighting_aug is not None
    assert S.state_dicts_equal(p_off, p_on)   # same seed, PhotometricAug adds no parameters
    return cfg_off, p_off, p_on


def _loss(policy, batch, seed=0):
    torch.manual_seed(seed)
    return policy.compute_loss(copy.deepcopy(batch))


def test_t2a_disabled_when_use_aug_false():
    cfg, p_off, p_on = _pair(ALL_WIDE)
    batch = S.random_batch(cfg, B=4)
    p_off.use_aug = p_on.use_aug = False
    l_off, d_off = _loss(p_off, batch)
    l_on, d_on = _loss(p_on, batch)
    assert torch.equal(l_off, l_on)
    assert d_off == d_on
    assert not any(k.startswith('lighting_aug/') for k in d_on)


def test_t2b_active_when_use_aug_true():
    cfg, p_off, p_on = _pair(ALL_WIDE)
    batch = S.random_batch(cfg, B=4)
    p_off.use_aug = p_on.use_aug = True
    l_off, d_off = _loss(p_off, batch)
    l_on, d_on = _loss(p_on, batch)
    assert not torch.equal(l_off, l_on)
    assert torch.isfinite(l_on)
    assert d_on['lighting_aug/profile_identity_frac'] == 0.0
    assert d_on['lighting_aug/profile_wide_frac'] == 1.0
    for key in p_on.rgb_obs_keys:
        assert isinstance(d_on[f'lighting_aug/{key}/mean_abs_delta'], float)
        assert d_on[f'lighting_aug/{key}/mean_abs_delta'] > 0.0
    assert p_on.lighting_aug.last_profiles.shape == (4,)
    # stats are not carried into a later call with use_aug=False
    p_on.use_aug = False
    _, d_again = _loss(p_on, batch)
    assert not any(k.startswith('lighting_aug/') for k in d_again)


def test_t2c_state_dict_compatible_both_ways():
    _, p_off, p_on = _pair()
    assert list(p_off.state_dict().keys()) == list(p_on.state_dict().keys())
    p_on.load_state_dict(p_off.state_dict(), strict=True)
    p_off.load_state_dict(p_on.state_dict(), strict=True)


class _ZeroDenoiser(torch.nn.Module):
    """Stand-in denoiser: the baseline inpainting branch (obs_as_global_cond=False) cannot run any
    repo denoiser (ConditionalUnet1D raises UnboundLocalError on global_cond=None, MLPResNet needs
    state_dim), so T2d isolates the observation / lighting path that precedes the model call."""

    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        assert global_cond is None and local_cond is None
        return torch.zeros_like(sample)


def test_t2d_inpainting_branch_uses_frames_per_sample():
    # policy.obs_as_global_cond=False flattens every horizon frame (F6): 4 frames per sample with
    # n_obs_steps=1, so frames-per-sample must be derived from img.shape[0] // batch_size.
    # use_vib=True: the baseline inpainting branch only defines loss_items on the Recon_VIB_loss path.
    cfg = S.small_cfg(use_vib=True, use_recon=False, extra=[
        'policy.obs_as_global_cond=false', 'policy.model=dp3', 'horizon=4', 'n_obs_steps=1', 'n_action_steps=4',
        'policy.lighting_aug.enabled=true', *ALL_WIDE])
    policy = S.build_policy(cfg, seed=0)
    assert not policy.obs_as_global_cond
    policy.model = _ZeroDenoiser()
    B = 2
    batch = S.random_batch(cfg, B=B, n_frames=cfg.horizon)
    policy.use_aug = True
    loss, loss_dict = _loss(policy, batch)
    assert torch.isfinite(loss)
    assert loss_dict['lighting_aug/profile_wide_frac'] == 1.0
    assert policy.lighting_aug.last_profiles.shape == (B,)   # per-sample, not per-frame
    for key in policy.rgb_obs_keys:
        assert loss_dict[f'lighting_aug/{key}/mean_abs_delta'] > 0.0
    # same batch, lighting off -> different loss (the aug really reached the encoder input)
    policy.lighting_aug = None
    loss_off, d_off = _loss(policy, batch)
    assert not torch.equal(loss, loss_off)
    assert not any(k.startswith('lighting_aug/') for k in d_off)
