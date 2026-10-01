"""T1: PhotometricAug unit tests (spec §8.2). Pure torch, no Hydra."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _support  # noqa: F401,E402  (puts RL-100 on sys.path)

from rl_100.model.common.aug import RandomShiftsAug  # noqa: E402
from rl_100.model.common.photometric_aug import PhotometricAug  # noqa: E402

CFG = {
    'enabled': True,
    'profile_prob': {'identity': 0.25, 'mild': 0.50, 'wide': 0.25},
    'white_balance_p': 0.5,
    'contrast_p': 0.3,
    'smooth_field_p_given_wide': 0.2,
    'share_profile_across_views': True,
    'share_params_across_history': True,
    'field_grid': 4,
    'mild': {'gain': [0.85, 1.15], 'gamma': [0.90, 1.10], 'rgb_gain': [0.95, 1.05], 'contrast': [0.90, 1.10]},
    'wide': {'gain': [0.70, 1.40], 'gamma': [0.80, 1.25], 'rgb_gain': [0.90, 1.10], 'contrast': [0.80, 1.20],
             'smooth_field': [0.85, 1.15]},
    'log_stats': True,
}
B, N, H, W = 8, 3, 24, 20   # non-square so layout mix-ups fail loudly


def _frames(seed=0, nhwc=False):
    x = torch.rand(B * N, 3, H, W, generator=torch.Generator().manual_seed(seed))
    return x.permute(0, 2, 3, 1).contiguous() if nhwc else x


def _const_frames(seed=0):
    """Every frame of a sample is the same constant colour (different across samples)."""
    g = torch.Generator().manual_seed(seed)
    colour = torch.rand(B, 1, 3, 1, 1, generator=g)
    return colour.expand(B, N, 3, H, W).reshape(B * N, 3, H, W).contiguous()


def test_t1a_identity_is_bit_exact():
    m = PhotometricAug(CFG)
    for nhwc in (False, True):
        x = _frames(nhwc=nhwc)
        out, stats = m({'cam': x.clone()}, ['cam'], B, profile='identity')
        assert torch.equal(out['cam'], x)
        assert out['cam'].dtype == torch.float32
        assert stats['profile_identity_frac'] == 1.0
        assert stats['cam/mean_abs_delta'] == 0.0


def test_t1b_history_frames_share_parameters():
    m = PhotometricAug({**CFG, 'smooth_field_p_given_wide': 1.0})
    x = _const_frames()
    out, _ = m({'cam': x.clone()}, ['cam'], B, profile='wide', generator=torch.Generator().manual_seed(1))
    y = out['cam'].view(B, N, 3, H, W)
    for b in range(B):
        for t in range(1, N):
            assert torch.equal(y[b, 0], y[b, t])


def test_t1c_samples_and_cameras_are_independent():
    m = PhotometricAug(CFG)
    x = _const_frames()
    out, _ = m({'cam': x.clone()}, ['cam'], B, profile='wide', generator=torch.Generator().manual_seed(2))
    y = out['cam'].view(B, N, 3, H, W)[:, 0].reshape(B, -1)
    assert not all(torch.equal(y[0], y[b]) for b in range(1, B))
    g = torch.Generator().manual_seed(3)
    profiles = torch.full((B,), 2, dtype=torch.long)
    p_cam0 = m.sample_params(profiles, 'cpu', generator=g)   # consecutive calls == consecutive cameras
    p_cam1 = m.sample_params(profiles, 'cpu', generator=g)
    assert not torch.equal(p_cam0['gain'] * p_cam0['gamma'], p_cam1['gain'] * p_cam1['gamma'])


def test_t1d_profiles_shared_across_cameras():
    m = PhotometricAug(CFG)
    obs = {'a': _frames(1), 'b': _frames(2), 'agent_pos': torch.zeros(B * N, 7)}
    m(obs, ['a', 'b', 'missing_key'], B, generator=torch.Generator().manual_seed(4))
    assert m.last_profiles.shape == (B,)
    assert m.last_profiles.dtype == torch.long
    assert torch.equal(obs['agent_pos'], torch.zeros(B * N, 7))


@pytest.mark.parametrize('nhwc', [False, True])
def test_t1e_clamped_layout_and_dtype(nhwc):
    m = PhotometricAug({**CFG, 'smooth_field_p_given_wide': 1.0, 'white_balance_p': 1.0, 'contrast_p': 1.0})
    x = _frames(5, nhwc=nhwc)
    x = (x * 1.6 - 0.3).clamp(0, 1)          # push mass towards both saturation ends
    out, stats = m({'cam': x.clone()}, ['cam'], B, profile='wide', generator=torch.Generator().manual_seed(6))
    y = out['cam']
    assert y.shape == x.shape and y.dtype == torch.float32
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    assert not torch.equal(y, x)
    for key in ('cam/sat_dark_delta', 'cam/sat_bright_delta', 'cam/mean_abs_delta',
                'profile_identity_frac', 'profile_mild_frac', 'profile_wide_frac'):
        assert isinstance(stats[key], float)


def test_t1f_stateless_module():
    m = PhotometricAug(CFG)
    assert len(list(m.parameters())) == 0
    assert len(list(m.buffers())) == 0
    assert len(m.state_dict()) == 0


def test_t1g_generator_reproducible_and_global_rng_untouched():
    m = PhotometricAug(CFG)
    x = _frames(7)
    o1, s1 = m({'cam': x.clone()}, ['cam'], B, generator=torch.Generator().manual_seed(9))
    o2, s2 = m({'cam': x.clone()}, ['cam'], B, generator=torch.Generator().manual_seed(9))
    assert torch.equal(o1['cam'], o2['cam']) and s1 == s2
    torch.manual_seed(123); ref = torch.rand(4)
    torch.manual_seed(123); m({'cam': x.clone()}, ['cam'], B, generator=torch.Generator().manual_seed(9))
    assert torch.equal(torch.rand(4), ref)   # explicit generator => global RNG not consumed
    # RandomShiftsAug: generator kwarg works, default path unchanged
    aug = RandomShiftsAug(pad=4)
    y_gen = aug(x, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(5); y_a = aug(x)
    torch.manual_seed(5); y_b = aug(x)
    assert y_gen.shape == y_a.shape == (B * N, H, W, 3)
    assert torch.equal(y_a, y_b)
    torch.manual_seed(123); aug(x, generator=torch.Generator().manual_seed(1))
    assert torch.equal(torch.rand(4), ref)


def test_t1h_config_validation():
    with pytest.raises(ValueError):
        PhotometricAug({**CFG, 'white_balance_p': 1.5})
    with pytest.raises(ValueError):
        PhotometricAug({**CFG, 'mild': {**CFG['mild'], 'gain': [1.2, 0.8]}})
    with pytest.raises(ValueError):
        PhotometricAug({**CFG, 'profile_prob': {'identity': 0.0, 'mild': 0.0, 'wide': 0.0}})
    with pytest.raises(NotImplementedError):
        PhotometricAug({**CFG, 'share_profile_across_views': False})
    with pytest.raises(NotImplementedError):
        PhotometricAug({**CFG, 'share_params_across_history': False})
    m = PhotometricAug({**CFG, 'profile_prob': {'identity': 1, 'mild': 1, 'wide': 2}})
    assert m.profile_probs == pytest.approx([0.25, 0.25, 0.5])
    with pytest.raises(ValueError):
        m.sample_profiles(B, 'cpu', profile='dark')
    with pytest.raises(ValueError):
        m.apply(_frames(), m.sample_params(torch.zeros(B, dtype=torch.long), 'cpu'), B + 1, torch.zeros(B + 1, dtype=torch.long))
