"""T3: MultiImageObsEncoder backbone / head split used by the online auxiliary (spec §8.4)."""
import hydra
import pytest
import torch

import _support as S


def _setup(use_vib, use_recon):
    cfg = S.small_cfg(use_vib=use_vib, use_recon=use_recon)
    policy = S.build_policy(cfg, seed=0)
    x = policy.obs2this_nobs(S.random_batch(cfg, B=4)['obs'], training=False)   # flattened (B*To, ...)
    return cfg, policy, policy.obs_encoder, x


@pytest.mark.parametrize('use_vib', [False, True])
def test_t3a_split_matches_forward(use_vib):
    _, policy, enc, x = _setup(use_vib=use_vib, use_recon=False)
    enc.force_stochastic = True            # online setting: the context manager must neutralise it
    with policy._deterministic_obs_encoder():
        ref = enc(x, deterministic=True)
        split = enc.head_forward(enc.backbone_forward(x), x)
    assert ref.shape == split.shape
    assert torch.allclose(ref, split, atol=1e-6)
    assert enc.force_stochastic is True    # restored on exit
    assert set(enc.backbone_forward(x).keys()) == set(enc.rgb_keys)


@pytest.mark.parametrize('use_vib,use_recon', [(True, False), (False, False), (True, True)])
def test_t3b_freeze_backbone(use_vib, use_recon):
    _, policy, enc, x = _setup(use_vib=use_vib, use_recon=use_recon)
    backbone = list(enc.backbone_parameters())
    assert all(p.requires_grad for p in backbone)
    n = enc.freeze_backbone()
    assert n == len(backbone) > 0
    assert all(not p.requires_grad for p in enc.backbone_parameters())
    assert all(not m.training for m in enc.key_model_map.modules())
    if use_vib:
        assert all(p.requires_grad for p in enc.vib_heads.parameters())
    if use_recon:
        assert all(p.requires_grad for p in enc.decoders.parameters())
    assert enc.has_trainable_params() is (use_vib or use_recon)
    # frozen backbone still runs and the split still matches forward()
    with policy._deterministic_obs_encoder():
        assert torch.allclose(enc(x, deterministic=True), enc.head_forward(enc.backbone_forward(x), x), atol=1e-6)


def test_t3c_shared_rgb_model_not_supported():
    cfg = S.small_cfg(use_vib=False, use_recon=False, extra=['encoders.resnet.share_rgb_model=true'])
    torch.manual_seed(0)
    enc = hydra.utils.instantiate(cfg.encoders.resnet)
    assert enc.share_rgb_model
    meta = S.shape_meta(cfg)
    x = {k: torch.rand(2, *attr['shape']) for k, attr in meta['obs'].items()}
    for call in (lambda: enc.backbone_forward(x), enc.freeze_backbone, enc.has_trainable_params,
                 lambda: list(enc.backbone_parameters()), lambda: enc.head_forward({}, x)):
        with pytest.raises(NotImplementedError):
            call()
    assert enc(x).shape[0] == 2   # forward itself is untouched
