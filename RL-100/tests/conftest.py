"""pytest fixtures for the lighting-aug suite (spec §8.1). Fixtures are factories so a test
can build several configs / policies with different switches."""
import os
import sys

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import _support as S  # noqa: E402  (installs import stubs, registers the eval resolver)


@pytest.fixture(scope='session')
def small_cfg():
    """small_cfg(use_vib, use_recon, fix_encoder=None, freeze_rgb_backbone=None, extra=()) -> DictConfig"""
    return S.small_cfg


@pytest.fixture(scope='session')
def policy():
    """policy(cfg, seed=0) -> RL1002D with a fitted LinearNormalizer, in eval mode"""
    return S.build_policy


@pytest.fixture(scope='session')
def random_batch():
    """random_batch(cfg, B=4, seed=1, n_frames=None) -> {'obs': ..., 'action': ...}"""
    return S.random_batch


@pytest.fixture(scope='session')
def ppo():
    """ppo(cfg, policy) -> BehaviorProximalPolicyOptimization after transfer2online"""
    return S.build_ppo


@pytest.fixture(scope='session')
def fake_replay():
    """fake_replay(cfg, B=None, seed=7) -> object with numpy_to_tensor()"""
    return S.FakeReplay
