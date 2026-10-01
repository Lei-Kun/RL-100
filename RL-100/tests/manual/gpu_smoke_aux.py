#!/usr/bin/env python
"""GPU smoke test for the lighting-aug online auxiliary (run on a machine with CUDA).

The CPU test-suite cannot exercise the CUDA-specific paths:
  torch.Generator(device='cuda') with randperm / rand / multinomial / randint,
  torch.random.fork_rng(devices=[...]), RandomShiftsAug(generator=cuda_gen).
This script runs them once and repeats the lambda=0 bit-exactness comparison on the GPU.

    cd RL-100
    python tests/manual/gpu_smoke_aux.py            # default cuda:0

A mismatch in [3] only matters if [2] passed (otherwise the GPU kernels themselves are
non-deterministic for this config and bit-exactness cannot be judged on this device).
"""
import argparse
import copy
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import _support as S  # noqa: E402

import torch  # noqa: E402


def _cfg(device, use_vib, fix_encoder, freeze, aux_kw=None):
    ov = [o for o in S.BASE_OVERRIDES if not o.startswith('training.device=')]
    ov += [f'training.device={device}', f'use_vib={str(use_vib).lower()}', 'use_recon=false',
           f'ppo.fix_encoder={str(fix_encoder).lower()}', f'ppo.freeze_rgb_backbone={str(freeze).lower()}']
    if aux_kw is not None:
        ov.append('ppo.aux_consistency.enabled=true')
        ov += [f'ppo.aux_consistency.{k}={v}' for k, v in aux_kw.items()]
    return S.compose_cfg(ov)


class _DeviceReplay:
    def __init__(self, replay, device):
        self._items = [
            ({k: v.to(device) for k, v in x.items()} if isinstance(x, dict) else x.to(device))
            for x in replay.numpy_to_tensor()
        ]

    def numpy_to_tensor(self):
        return tuple(self._items)


def _update(cfg, policy, replay, seed=0):
    ppo = S.build_ppo(cfg, copy.deepcopy(policy), seed=seed)
    torch.manual_seed(seed)
    out = ppo.dp_align_update_no_share(replay, total_steps=1)
    return ppo, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    assert torch.cuda.is_available(), 'this smoke test needs a CUDA device'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = args.device

    # [1] every aux mode runs on the GPU
    for use_vib, fix_encoder, freeze, expect in [(True, True, False, 'cached_full'),
                                                 (True, False, True, 'cached_backbone'),
                                                 (False, False, True, 'cached_full'),
                                                 (True, False, False, 'full')]:
        cfg = _cfg(dev, use_vib, fix_encoder, freeze, aux_kw=dict(warmup_steps=0, grad_log_every=1,
                                                                 log_post_update_kl='true'))
        policy = S.build_policy(cfg, seed=0)
        replay = _DeviceReplay(S.FakeReplay(cfg, policy=policy), dev)
        ppo, out = _update(cfg, policy, replay)
        m = ppo.last_aux_metrics
        ok = m.get(f'aux/mode_{expect}') == 1.0 and math.isfinite(m['aux/loss']) and all(math.isfinite(float(v)) for v in out)
        print(f"[1] use_vib={use_vib} fix_encoder={fix_encoder} freeze={freeze}: mode={expect} "
              f"aux/loss={m['aux/loss']:.3e} ppo_grad={m.get('aux/ppo_grad_norm', float('nan')):.3e} "
              f"aux_grad={m.get('aux/aux_grad_norm', float('nan')):.3e} kl={m.get('aux/post_update_approx_kl')} -> {'OK' if ok else 'FAIL'}")

    # [2] baseline determinism on this device (aux off, run twice)
    cfg_off = _cfg(dev, True, False, False)
    policy = S.build_policy(cfg_off, seed=0)
    replay = _DeviceReplay(S.FakeReplay(cfg_off, policy=policy), dev)
    p1, _ = _update(cfg_off, policy, replay)
    p2, _ = _update(cfg_off, policy, replay)
    det = S.state_dicts_equal(p1._policy, p2._policy)
    print(f"[2] baseline run-to-run bit-identical on {dev}: {det}")

    # [3] lambda=0 auxiliary vs aux off
    cfg_on = _cfg(dev, True, False, False, aux_kw=dict(lambda_actor=0.0, warmup_steps=0))
    p3, _ = _update(cfg_on, policy, replay)
    same = S.state_dicts_equal(p1._policy, p3._policy)
    print(f"[3] aux(lambda=0) vs aux off bit-identical on {dev}: {same}"
          + ('' if det else '   (inconclusive: [2] failed)'))


if __name__ == '__main__':
    main()
