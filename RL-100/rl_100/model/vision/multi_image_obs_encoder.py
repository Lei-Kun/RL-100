from typing import Dict, Tuple, Union
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from rl_100.model.vision.crop_randomizer import CropRandomizer
from rl_100.model.common.module_attr_mixin import ModuleAttrMixin
from rl_100.common.pytorch_util import dict_apply, replace_submodules
from copy import deepcopy
import einops
class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            use_group_norm: bool=False,
            share_rgb_model: bool=False,
            imagenet_norm: bool=False,
            use_agent_pos: bool=True,
            use_vib: bool=False,
            use_recon: bool=False,
            recon_loss_weight: float=0.05,
            kl_loss_weight: float=1.,
            kl_beta: float=2.5e-4,
            latent_dim: int=None,
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_shape_map = dict()
        self.use_agent_pos = use_agent_pos
        self.recon_loss_weight = recon_loss_weight
        self.kl_loss_weight = kl_loss_weight
        # Fix 5: unified naming for KL annealing
        self.kl_beta = kl_beta
        self.beta_kl = kl_beta
        # Fix 7: force_stochastic for online finetuning
        self.force_stochastic = False
        self.use_vib = use_vib
        self.use_recon = use_recon
        # Fix 6: optional bottleneck dim
        self.latent_dim = latent_dim
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.imagenet_norm = imagenet_norm
        self.debug_image_norm = os.environ.get("DEBUG_IMAGE_NORM", "0") == "1"
        self._debug_image_norm_seen = set()

        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        # Fix 2/3: split transforms into spatial (resize+crop) and normalize
        key_spatial_transform_map = nn.ModuleDict()
        key_normalize_map = nn.ModuleDict()

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        this_model = copy.deepcopy(rgb_model)
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16,
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model

                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h,w))
                    input_shape = (shape[0],h,w)

                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h, crop_width=w,
                            num_crops=1, pos_enc=False)
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(h,w))

                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                # Fix 2/3: only keep spatial + normalize as separate maps
                this_spatial = nn.Sequential(this_resizer, this_randomizer)
                key_spatial_transform_map[key] = this_spatial
                key_normalize_map[key] = this_normalizer
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                continue

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_spatial_transform_map = key_spatial_transform_map
        self.key_normalize_map = key_normalize_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        # First pass: compute _last_feature_dims WITHOUT VIB (heads don't exist yet)
        _saved_use_vib = self.use_vib
        self.use_vib = False
        self.output_shape()
        self.use_vib = _saved_use_vib

        # VIB head setup (Fix 6: support latent_dim bottleneck)
        if self.use_vib:
            if self.share_rgb_model:
                img_feat_dim = self._last_feature_dims[self.rgb_keys[0]]
                img_latent_dim = latent_dim if latent_dim is not None else img_feat_dim
                self.vib_head = nn.ModuleDict({
                    'mu': nn.Linear(img_feat_dim, img_latent_dim),
                    'logvar': nn.Linear(img_feat_dim, img_latent_dim)
                })
            else:
                self.vib_heads = nn.ModuleDict()
                for key in self.rgb_keys:
                    feat_dim = self._last_feature_dims[key]
                    ld = latent_dim if latent_dim is not None else feat_dim
                    self.vib_heads[key] = nn.ModuleDict({
                        'mu': nn.Linear(feat_dim, ld),
                        'logvar': nn.Linear(feat_dim, ld)
                    })

        # decoder setup (Fix 6: decoder input uses latent_dim)
        if self.use_recon:
            if self.share_rgb_model:
                feat_dim = self._last_feature_dims[self.rgb_keys[0]]
                dec_input_dim = latent_dim if (latent_dim is not None and self.use_vib) else feat_dim
                self.decoder = self.build_decoder(self.rgb_keys[0], latent_dim=dec_input_dim)
            else:
                self.decoders = nn.ModuleDict()
                for key in self.rgb_keys:
                    feat_dim = self._last_feature_dims[key]
                    dec_input_dim = latent_dim if (latent_dim is not None and self.use_vib) else feat_dim
                    self.decoders[key] = self.build_decoder(key, latent_dim=dec_input_dim)

    def build_decoder(self, key, latent_dim):
        """
        Build a decoder that reconstructs images of shape (C, H0, W0).
        """
        import math
        attr = self.shape_meta['obs'][key]
        C, H0, W0 = attr['shape']
        if self.resize_shape:
            H0, W0 = (self.resize_shape[key]
                    if isinstance(self.resize_shape, dict)
                    else self.resize_shape)
        if self.crop_shape:
            H0, W0 = (self.crop_shape[key]
                    if isinstance(self.crop_shape, dict)
                    else self.crop_shape)

        sf_h = H0 / 4
        sf_w = W0 / 4
        use_pure = False
        if sf_h == sf_w and sf_h > 0 and float(sf_h).is_integer():
            log2_sf = math.log2(sf_h)
            if float(log2_sf).is_integer():
                use_pure = True
                n_layers = int(log2_sf)

        layers = [nn.Linear(latent_dim, 64 * 4 * 4), nn.ReLU(), nn.Unflatten(1, (64, 4, 4))]

        if use_pure:
            in_ch = 64
            for i in range(n_layers):
                out_ch = max(in_ch // 2, C)
                layers += [
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.ReLU()
                ]
                in_ch = out_ch
            if in_ch != C:
                layers += [nn.Conv2d(in_ch, C, kernel_size=1), nn.ReLU()]
            layers += [nn.Sigmoid()]
        else:
            layers += [
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(8, C, 4, stride=2, padding=1),
                nn.Sigmoid(),
                nn.Upsample(size=(H0, W0), mode='bilinear', align_corners=False)
            ]
        return nn.Sequential(*layers)

    def _apply_transform(self, key, img, deterministic=False):
        """Apply full transform (spatial + normalize) to image."""
        self._log_image_stats("input_raw", key, img)
        img = self._apply_spatial_transform(key, img, deterministic=deterministic)
        self._log_image_stats("after_spatial", key, img)
        img = self.key_normalize_map[key](img)
        self._log_image_stats("after_imagenet_norm", key, img)
        return img

    def _apply_spatial_transform(self, key, img, deterministic=False):
        """Apply only spatial transform (resize+crop), no normalization. For recon targets."""
        # Auto-detect [0,255] uint8-origin images and normalize to [0,1]
        if img.max() > 1.0:
            img = img / 255.0
        if deterministic:
            was_training = self.key_spatial_transform_map[key].training
            self.key_spatial_transform_map[key].eval()
            img = self.key_spatial_transform_map[key](img)
            if was_training:
                self.key_spatial_transform_map[key].train()
        else:
            img = self.key_spatial_transform_map[key](img)
        return img

    def _log_image_stats(self, stage, key, tensor):
        if not self.debug_image_norm:
            return
        if not torch.is_tensor(tensor):
            return
        debug_key = (stage, key)
        if debug_key in self._debug_image_norm_seen:
            return
        self._debug_image_norm_seen.add(debug_key)
        with torch.no_grad():
            t = tensor.detach().float()
            print(
                f"[DEBUG_IMAGE_NORM] stage={stage} key={key} "
                f"shape={tuple(t.shape)} min={t.min().item():.4f} "
                f"max={t.max().item():.4f} mean={t.mean().item():.4f} "
                f"std={t.std().item():.4f}",
                flush=True,
            )

    def _vib_forward(self, feature, head):
        """Apply VIB head: train/force_stochastic -> sample, eval -> mu."""
        mu = head['mu'](feature)
        logvar = head['logvar'](feature)
        if self.training or self.force_stochastic:
            std = torch.exp(0.5 * logvar)
            feature = mu + std * torch.randn_like(std)
        else:
            feature = mu
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return feature, kl_loss

    # ------------------------------------------------------------------ backbone / head split
    # Used by the online prediction-consistency auxiliary (uni_ppo.py). Additive only:
    # forward / Recon_VIB_loss / output_shape are untouched.
    def _require_per_key_models(self):
        if self.share_rgb_model:
            raise NotImplementedError('share_rgb_model=True is not supported')

    def backbone_parameters(self):
        """Parameters of the per-camera rgb backbones only (no VIB heads, no decoders)."""
        self._require_per_key_models()
        return self.key_model_map.parameters()

    def freeze_backbone(self):
        """Freeze the rgb backbones in place; returns the number of frozen tensors.
        Backbones are also switched to eval() (no BN running buffers exist, see F7)."""
        self._require_per_key_models()
        n = 0
        for p in self.backbone_parameters():
            p.requires_grad_(False)
            p.grad = None
            n += 1
        self.key_model_map.eval()
        return n

    def has_trainable_params(self):
        """True if any non-empty parameter still requires grad (ignores ModuleAttrMixin's
        zero-element _dummy_variable, which is never optimised)."""
        self._require_per_key_models()
        return any(p.requires_grad and p.numel() > 0 for p in self.parameters())

    def backbone_forward(self, obs_dict):
        """rgb key -> backbone feature (B*n, D_key), including _apply_transform(deterministic=True).
        No VIB, no low_dim."""
        self._require_per_key_models()
        feats = {}
        for key in self.rgb_keys:
            img = obs_dict[key]
            if img.shape[1] != 3:
                img = einops.rearrange(img, 'b h w c -> b c h w')
            assert img.shape[1:] == self.key_shape_map[key]
            img = self._apply_transform(key, img, deterministic=True)
            feats[key] = self.key_model_map[key](img)
        return feats

    def head_forward(self, backbone_feats, obs_dict):
        """Same concatenation order as forward(): rgb_keys (after VIB) -> low_dim_keys.
        VIB determinism is the caller's responsibility (RL1002D._deterministic_obs_encoder)."""
        self._require_per_key_models()
        features = []
        batch_size = None
        for key in self.rgb_keys:
            f = backbone_feats[key]
            batch_size = f.shape[0]
            if self.use_vib:
                f, _ = self._vib_forward(f, self.vib_heads[key])
            features.append(f)
        if self.use_agent_pos:
            for key in self.low_dim_keys:
                data = obs_dict[key]
                assert data.shape[0] == batch_size and data.shape[1:] == self.key_shape_map[key]
                features.append(data.to(features[0].device) if features else data)
        return torch.cat(features, dim=-1)

    def forward(self, obs_dict, deterministic=False):
        batch_size = None
        features = list()
        self._last_feature_dims = {}
        # process rgb input
        if self.share_rgb_model:
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if img.shape[1] != 3:
                    img = einops.rearrange(img, "b h w c -> b c h w")
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self._apply_transform(key, img, deterministic=deterministic)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            feature = self.key_model_map['rgb'](imgs)
            D = feature.shape[-1]
            for key in self.rgb_keys:
                self._last_feature_dims[key] = D
            # Fix 1: apply VIB in forward() for shared model
            if self.use_vib:
                feature, _ = self._vib_forward(feature, self.vib_head)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            feature = torch.moveaxis(feature, 0, 1)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                if img.shape[1] != 3:
                    img = einops.rearrange(img, "b h w c -> b c h w")
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self._apply_transform(key, img, deterministic=deterministic)
                feature = self.key_model_map[key](img)
                self._last_feature_dims[key] = feature.shape[-1]
                # Fix 1: apply VIB in forward() for per-key model
                if self.use_vib:
                    feature, _ = self._vib_forward(feature, self.vib_heads[key])
                features.append(feature)

        # process lowdim input
        if self.use_agent_pos:
            for key in self.low_dim_keys:
                data = obs_dict[key]
                if batch_size is None:
                    batch_size = data.shape[0]
                else:
                    assert batch_size == data.shape[0]
                assert data.shape[1:] == self.key_shape_map[key]
                self._last_feature_dims[key] = data.shape[-1]
                # Ensure low_dim data is on the same device as RGB features
                if len(features) > 0:
                    data = data.to(features[0].device)
                features.append(data)

        result = torch.cat(features, dim=-1)
        return result

    def Recon_VIB_loss(self, obs_dict, deterministic=False):
        batch_size = None
        features = list()
        kl_losses   = []
        recon_losses = []
        recon_imgs   = {}
        self._last_feature_dims = {}
        # process rgb input
        if self.share_rgb_model:
            imgs = list()
            target_imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if img.shape[1] != 3:
                    img = einops.rearrange(img, "b h w c -> b c h w")
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                # Fix 2/3: apply spatial transform (same crop for both),
                # then split into normalized (encoder) and [0,1] (recon target)
                spatial_img = self._apply_spatial_transform(key, img, deterministic=deterministic)
                self._log_image_stats("recon_target", key, spatial_img)
                target_imgs.append(spatial_img.detach())  # [0,1] range after _apply_spatial_transform
                normalized_img = self.key_normalize_map[key](spatial_img)
                self._log_image_stats("recon_after_imagenet_norm", key, normalized_img)
                imgs.append(normalized_img)
            imgs = torch.cat(imgs, dim=0)
            feature = self.key_model_map['rgb'](imgs)
            D = feature.shape[-1]
            for key in self.rgb_keys:
                self._last_feature_dims[key] = D

            if self.use_vib:
                feature, kl_loss = self._vib_forward(feature, self.vib_head)
                kl_losses.append(kl_loss)

            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            if self.use_recon:
                for idx, key in enumerate(self.rgb_keys):
                    recon_img = self.decoder(feature[idx])
                    recon_imgs[key] = recon_img
                    recon_losses.append(F.mse_loss(recon_img, target_imgs[idx]))

            feature = torch.moveaxis(feature, 0, 1)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                if img.shape[1] != 3:
                    img = einops.rearrange(img, "b h w c -> b c h w")
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                # Fix 2/3: same crop, split normalize
                spatial_img = self._apply_spatial_transform(key, img, deterministic=deterministic)
                self._log_image_stats("recon_target", key, spatial_img)
                target_img = spatial_img.detach()
                normalized_img = self.key_normalize_map[key](spatial_img)
                self._log_image_stats("recon_after_imagenet_norm", key, normalized_img)
                feature = self.key_model_map[key](normalized_img)
                self._last_feature_dims[key] = feature.shape[-1]

                if self.use_vib:
                    feature, kl_loss = self._vib_forward(feature, self.vib_heads[key])
                    kl_losses.append(kl_loss)

                if self.use_recon:
                    recon_img = self.decoders[key](feature)
                    recon_imgs[key] = recon_img
                    recon_losses.append(F.mse_loss(recon_img, target_img))
                features.append(feature)

        # process lowdim input
        if self.use_agent_pos:
            for key in self.low_dim_keys:
                data = obs_dict[key]
                if batch_size is None:
                    batch_size = data.shape[0]
                else:
                    assert batch_size == data.shape[0]
                assert data.shape[1:] == self.key_shape_map[key]
                self._last_feature_dims[key] = data.shape[-1]
                # Ensure low_dim data is on the same device as RGB features
                if len(features) > 0:
                    data = data.to(features[0].device)
                features.append(data)

        loss = 0
        aux_losses = {}
        if self.use_recon:
            recon_loss = torch.stack(recon_losses).mean() * self.recon_loss_weight
            loss += recon_loss
            aux_losses['recon_loss'] = recon_loss.item()
        if self.use_vib:
            # Fix 5: use self.beta_kl
            vib_loss = torch.stack(kl_losses).mean() * self.kl_loss_weight * self.beta_kl
            loss += vib_loss
            aux_losses['kl_loss'] = vib_loss.item()
        result = torch.cat(features, dim=-1)
        return loss, aux_losses, result


    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape[-1]
