"""Photometric (lighting) augmentation for [0,1] float rgb frames.

Stateless ``nn.Module`` (no parameters, no buffers) so that adding it to a
policy does not change ``state_dict`` keys.  All randomness can be routed
through an explicit ``torch.Generator`` (``generator=None`` falls back to the
global RNG, which is what the offline SFT path uses).

Frame layout contract (see spec F5): the input is a flattened batch of
``batch_size * n`` frames where frame ``b * n + t`` belongs to sample ``b``.
Per-sample parameters are expanded with ``repeat_interleave(n, dim=0)`` so all
frames of one sample (its observation history) share the same lighting.
"""
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_MILD = {'gain': [0.85, 1.15], 'gamma': [0.90, 1.10],
                 'rgb_gain': [0.95, 1.05], 'contrast': [0.90, 1.10]}
_DEFAULT_WIDE = {'gain': [0.70, 1.40], 'gamma': [0.80, 1.25],
                 'rgb_gain': [0.90, 1.10], 'contrast': [0.80, 1.20],
                 'smooth_field': [0.85, 1.15]}
_DEFAULT_PROFILE_PROB = {'identity': 0.25, 'mild': 0.50, 'wide': 0.25}


def _check_prob(value, name: str) -> float:
    p = float(value)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f'PhotometricAug: {name} must be in [0, 1], got {value}')
    return p


def _check_range(value, name: str) -> Tuple[float, float]:
    if value is None or len(value) != 2:
        raise ValueError(f'PhotometricAug: {name} must be a [lo, hi] pair, got {value}')
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        raise ValueError(f'PhotometricAug: {name} must satisfy lo <= hi, got {value}')
    return lo, hi


class PhotometricAug(nn.Module):
    """Per-sample gain / gamma / white-balance / contrast / low-frequency field."""

    PROFILES = ('identity', 'mild', 'wide')   # index 0, 1, 2

    def __init__(self, cfg: dict):
        super().__init__()
        cfg = dict(cfg) if cfg is not None else {}
        self.cfg = cfg

        profile_prob = dict(cfg.get('profile_prob', _DEFAULT_PROFILE_PROB))
        unknown = set(profile_prob) - set(self.PROFILES)
        if unknown:
            raise ValueError(f'PhotometricAug: unknown profiles in profile_prob: {sorted(unknown)}')
        raw = [float(profile_prob.get(name, 0.0)) for name in self.PROFILES]
        if any(p < 0.0 for p in raw) or sum(raw) <= 0.0:
            raise ValueError(f'PhotometricAug: profile_prob must be non-negative with positive sum, got {profile_prob}')
        total = sum(raw)
        self.profile_probs = [p / total for p in raw]   # plain python list, not a buffer

        self.white_balance_p = _check_prob(cfg.get('white_balance_p', 0.5), 'white_balance_p')
        self.contrast_p = _check_prob(cfg.get('contrast_p', 0.3), 'contrast_p')
        self.smooth_field_p_given_wide = _check_prob(
            cfg.get('smooth_field_p_given_wide', 0.2), 'smooth_field_p_given_wide')

        if not bool(cfg.get('share_profile_across_views', True)):
            raise NotImplementedError('PhotometricAug v1 only supports share_profile_across_views=True')
        if not bool(cfg.get('share_params_across_history', True)):
            raise NotImplementedError('PhotometricAug v1 only supports share_params_across_history=True')

        self.field_grid = int(cfg.get('field_grid', 4))
        if self.field_grid < 1:
            raise ValueError(f'PhotometricAug: field_grid must be >= 1, got {self.field_grid}')

        mild = dict(_DEFAULT_MILD); mild.update(dict(cfg.get('mild', {}) or {}))
        wide = dict(_DEFAULT_WIDE); wide.update(dict(cfg.get('wide', {}) or {}))
        self.mild = {k: _check_range(mild[k], f'mild.{k}') for k in ('gain', 'gamma', 'rgb_gain', 'contrast')}
        self.wide = {k: _check_range(wide[k], f'wide.{k}') for k in ('gain', 'gamma', 'rgb_gain', 'contrast', 'smooth_field')}

        self.log_stats = bool(cfg.get('log_stats', True))
        self.last_profiles: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ sampling
    def sample_profiles(self, batch_size: int, device, profile: Optional[str] = None,
                        generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Return a LongTensor [B] of profile indices (0 identity, 1 mild, 2 wide)."""
        if batch_size <= 0:
            raise ValueError(f'PhotometricAug: batch_size must be positive, got {batch_size}')
        if profile is not None:
            if profile not in self.PROFILES:
                raise ValueError(f'PhotometricAug: unknown profile {profile!r}, expected one of {self.PROFILES}')
            return torch.full((batch_size,), self.PROFILES.index(profile), dtype=torch.long, device=device)
        probs = torch.tensor(self.profile_probs, dtype=torch.float32, device=device)
        return torch.multinomial(probs, batch_size, replacement=True, generator=generator)

    def _uniform_by_profile(self, u: torch.Tensor, wide_mask: torch.Tensor, key: str) -> torch.Tensor:
        """Map U[0,1] draws to U[lo, hi] with the range chosen per row (wide vs mild)."""
        m_lo, m_hi = self.mild[key]
        w_lo, w_hi = self.wide[key]
        lo = torch.where(wide_mask, torch.full_like(u, w_lo), torch.full_like(u, m_lo))
        hi = torch.where(wide_mask, torch.full_like(u, w_hi), torch.full_like(u, m_hi))
        return lo + (hi - lo) * u

    def sample_params(self, profiles: torch.Tensor, device,
                      generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """Per-sample lighting parameters. Identity rows get neutral values.

        Random draws are always taken for the whole batch in a fixed order so
        that a seeded generator reproduces the same parameters.
        """
        B = int(profiles.shape[0])
        active = (profiles != 0)
        wide = (profiles == 2)
        wide_b = wide.view(B, 1)

        def rand(*shape):
            return torch.rand(*shape, device=device, generator=generator)

        use_gamma = rand(B) < 0.5
        gamma_u = rand(B)
        gain_u = rand(B)
        wb_on = rand(B) < self.white_balance_p
        rgb_u = rand(B, 3)
        contrast_on = rand(B) < self.contrast_p
        contrast_u = rand(B)
        field_draw = rand(B) < self.smooth_field_p_given_wide

        ones = torch.ones(B, device=device)
        gamma = torch.where(active & use_gamma, self._uniform_by_profile(gamma_u, wide, 'gamma'), ones)
        gain = torch.where(active & ~use_gamma, self._uniform_by_profile(gain_u, wide, 'gain'), ones)
        rgb_gain = torch.where((active & wb_on).view(B, 1),
                               self._uniform_by_profile(rgb_u, wide_b.expand(B, 3), 'rgb_gain'),
                               torch.ones(B, 3, device=device))
        contrast = torch.where(active & contrast_on, self._uniform_by_profile(contrast_u, wide, 'contrast'), ones)
        field_on = wide & field_draw
        field_lo, field_hi = self.wide['smooth_field']
        return {
            'gain': gain, 'gamma': gamma, 'rgb_gain': rgb_gain, 'contrast': contrast,
            'field_on': field_on, 'field_lo': field_lo, 'field_hi': field_hi,
        }

    # ------------------------------------------------------------------ apply
    @staticmethod
    def _is_nhwc(img: torch.Tensor) -> bool:
        return img.ndim == 4 and img.shape[1] != 3 and img.shape[-1] == 3

    @torch.no_grad()
    def apply(self, img: torch.Tensor, params: Dict[str, torch.Tensor], batch_size: int,
              profiles: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Apply per-sample parameters to flattened frames.

        img: (B*n, C, H, W) or (B*n, H, W, C), float in [0,1]. Output keeps the
        input layout, dtype float32, clamped to [0,1]. Identity rows are copied
        bit-exactly from the (float32) input.
        """
        if img.ndim != 4:
            raise ValueError(f'PhotometricAug: expected 4D image batch, got shape {tuple(img.shape)}')
        if batch_size <= 0 or img.shape[0] % batch_size != 0:
            raise ValueError(f'PhotometricAug: frames {img.shape[0]} not divisible by batch_size {batch_size}')
        if profiles.shape[0] != batch_size:
            raise ValueError(f'PhotometricAug: profiles has {profiles.shape[0]} rows, expected {batch_size}')
        n = img.shape[0] // batch_size
        img = img.float()
        if bool((profiles == 0).all()):
            return img   # nothing to do; same object, no arithmetic

        nhwc = self._is_nhwc(img)
        x_in = img.permute(0, 3, 1, 2) if nhwc else img
        C, H, W = x_in.shape[1:]

        def expand(p):
            return p.repeat_interleave(n, dim=0)

        gain = expand(params['gain']).view(-1, 1, 1, 1)
        gamma = expand(params['gamma']).view(-1, 1, 1, 1)
        rgb_gain = expand(params['rgb_gain']).view(-1, C, 1, 1)
        contrast = expand(params['contrast']).view(-1, 1, 1, 1)

        x = x_in * gain
        x = x.clamp(0.0, 1.0) ** gamma
        x = x * rgb_gain
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - m) * contrast + m

        field_on = params['field_on']
        if bool(field_on.any()):
            g = self.field_grid
            lo, hi = float(params['field_lo']), float(params['field_hi'])
            field = lo + (hi - lo) * torch.rand((batch_size, 1, g, g), device=x.device, generator=generator)
            field = F.interpolate(field, size=(H, W), mode='bilinear', align_corners=False)
            field = torch.where(field_on.view(-1, 1, 1, 1), field, torch.ones_like(field))
            x = x * expand(field)

        x = x.clamp(0.0, 1.0)
        identity_frame = expand(profiles == 0).view(-1, 1, 1, 1)
        out = torch.where(identity_frame, x_in, x)
        if nhwc:
            out = out.permute(0, 2, 3, 1)
        return out

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, obs_dict: dict, rgb_keys: Sequence[str], batch_size: int,
                profile: Optional[str] = None,
                generator: Optional[torch.Generator] = None) -> Tuple[dict, dict]:
        """Augment every rgb key present in ``obs_dict`` in place (new tensors).

        One profile draw per call is shared by all cameras; the continuous
        parameters are drawn independently per camera. Returns
        ``(obs_dict, stats)``; ``stats`` is empty when ``log_stats`` is False.
        """
        keys = [k for k in rgb_keys if k in obs_dict]
        if not keys:
            self.last_profiles = None
            return obs_dict, {}
        device = obs_dict[keys[0]].device
        profiles = self.sample_profiles(batch_size, device, profile=profile, generator=generator)
        self.last_profiles = profiles
        stats: Dict[str, float] = {}
        if self.log_stats:
            for idx, name in enumerate(self.PROFILES):
                stats[f'profile_{name}_frac'] = float((profiles == idx).float().mean().item())
        for key in keys:
            img = obs_dict[key]
            params = self.sample_params(profiles, device, generator=generator)
            out = self.apply(img, params, batch_size, profiles, generator=generator)
            if self.log_stats:
                inp = img.float()
                dark, bright = 1.0 / 255.0, 254.0 / 255.0
                stats[f'{key}/sat_dark_delta'] = float(
                    ((out <= dark).float().mean() - (inp <= dark).float().mean()).item())
                stats[f'{key}/sat_bright_delta'] = float(
                    ((out >= bright).float().mean() - (inp >= bright).float().mean()).item())
                stats[f'{key}/mean_abs_delta'] = float((out - inp).abs().mean().item())
            obs_dict[key] = out
        return obs_dict, stats
