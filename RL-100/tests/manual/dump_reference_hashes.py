#!/usr/bin/env python
"""Bit-exactness reference dump (spec §9.1).

Runs, under fixed seeds and with every new switch left at its yaml default (off):
  (a) one RL1002D.compute_loss (use_aug False and True) -> loss values + parameter SHA256
  (b) one BehaviorProximalPolicyOptimization.dp_align_update_no_share -> updated _policy SHA256
and writes a JSON record. Run it on the baseline and on the new code; `diff` must be empty.

    cd RL-100
    git stash
    python tests/manual/dump_reference_hashes.py --out /path/ref_base.json
    git stash pop
    python tests/manual/dump_reference_hashes.py --out /path/ref_new.json
    diff /path/ref_base.json /path/ref_new.json

The baseline has no policy.lighting_aug / ppo.aux_consistency keys, so this script never
overrides them (defaults must be equivalent to "absent").
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import _support as S  # noqa: E402

import torch  # noqa: E402


def _bool(v):
    return str(v).lower() in ('1', 'true', 'yes')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--use-vib', default='true')
    ap.add_argument('--use-recon', default='true')
    ap.add_argument('--fix-encoder', default='false')
    ap.add_argument('--config-name', default=S.DEFAULT_CONFIG_NAME)
    args = ap.parse_args()

    overrides = S.small_overrides(use_vib=_bool(args.use_vib), use_recon=_bool(args.use_recon),
                                  fix_encoder=_bool(args.fix_encoder))
    cfg = S.compose_cfg(overrides, config_name=args.config_name)
    rec = {'config_name': args.config_name, 'overrides': overrides, 'seed': args.seed,
           'torch_num_threads': torch.get_num_threads()}

    policy = S.build_policy(cfg, seed=args.seed)
    rec['policy_sha256_init'] = S.state_dict_sha256(policy)
    batch = S.random_batch(cfg, B=4, seed=1)
    for use_aug in (False, True):
        policy.use_aug = use_aug
        torch.manual_seed(args.seed)
        loss, loss_dict = policy.compute_loss(copy.deepcopy(batch))
        rec[f'compute_loss_use_aug_{str(use_aug).lower()}'] = {
            'loss': float(loss.item()), 'loss_dict': {k: float(v) for k, v in loss_dict.items()}}
    policy.use_aug = False

    torch.manual_seed(args.seed)
    ppo = S.build_ppo(cfg, policy)
    rec['optimizer_actor_num_tensors'] = sum(len(g['params']) for g in ppo.optimizer_actor.param_groups)
    replay = S.FakeReplay(cfg, seed=7)
    torch.manual_seed(args.seed + 1)
    out = ppo.dp_align_update_no_share(replay, total_steps=1)
    rec['dp_align_returns'] = [float(x) for x in out]
    rec['policy_sha256_after_update'] = S.state_dict_sha256(ppo._policy)
    rec['old_policy_sha256'] = S.state_dict_sha256(ppo._old_policy)
    rec['critic_sha256_after_update'] = S.state_dict_sha256(ppo.critic)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rec, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({k: rec[k] for k in ('policy_sha256_init', 'policy_sha256_after_update')}, indent=2))


if __name__ == '__main__':
    main()
