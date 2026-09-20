
__all__ = [
    'AdroitEnv',
    'DexArtEnv',
    'MetaWorldEnv',
    'MetaWorldMultiViewEnv',
    'make_dmc_env',
    'make_dmc_env_2d',
    'DMCEnv',
    'UR5Env',
    'FrankaEnv',
    'FrankaPourEnv',
    'PegEnv',
    'FlippingEnv',
]

def __getattr__(name):
    if name == 'AdroitEnv':
        from .adroit import AdroitEnv
        return AdroitEnv
    if name == 'DexArtEnv':
        from .dexart import DexArtEnv
        return DexArtEnv
    if name in ('MetaWorldEnv', 'MetaWorldMultiViewEnv'):
        from .metaworld import MetaWorldEnv, MetaWorldMultiViewEnv
        return {'MetaWorldEnv': MetaWorldEnv, 'MetaWorldMultiViewEnv': MetaWorldMultiViewEnv}[name]
    if name in ('make_dmc_env', 'DMCEnv', 'make_dmc_env_2d'):
        from .dmc import DMCEnv, make_dmc_env, make_dmc_env_2d
        return {
            'make_dmc_env': make_dmc_env,
            'DMCEnv': DMCEnv,
            'make_dmc_env_2d': make_dmc_env_2d,
        }[name]
    if name == 'UR5Env':
        from .ur5 import UR5Env
        return UR5Env
    if name == 'FrankaEnv':
        from .franka import FrankaEnv
        return FrankaEnv
    if name == 'FrankaPourEnv':
        from .franka_pour import FrankaPourEnv
        return FrankaPourEnv
    if name == 'PegEnv':
        from .peg import PegEnv
        return PegEnv
    if name == 'FlippingEnv':
        # Keep real-robot dependencies lazy so sim tasks importing rl_100.env do
        # not start flipping's keyboard listener in every eval worker.
        from .flipping import FlippingEnv
        return FlippingEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
