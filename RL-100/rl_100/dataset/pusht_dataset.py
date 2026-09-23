from typing import Dict
import torch
import numpy as np
import copy
from rl_100.common.pytorch_util import dict_apply
from rl_100.common.replay_buffer import ReplayBuffer
from rl_100.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from rl_100.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from rl_100.dataset.base_dataset import BaseDataset
from rl_100.unidpg.utils import RewardScaling
from rl_100.common.normalize_util import get_image_range_normalizer

from termcolor import cprint
from tqdm import tqdm
def compute_return(reward, not_done, gamma: float == 0.99
    ) -> np.ndarray:
        size_ = len(reward)
        return_ = np.zeros((size_, 1))
        pre_return = 0
        for i in tqdm(reversed(range(size_)), desc='Computing the returns'):
            return_[i] = reward[i] + gamma * pre_return * not_done[i]
            pre_return = return_[i]
        return return_
class PushTDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            scale_strategy=None,
            pre_image_norm=False,   
            use_img=False,
            image_keys=None,
            derive_next_image=False,
            derive_next_obs=False,
            sequence_stride=1,
            derive_next_action=False,
            ):
        super().__init__()
        self.task_name = task_name
        self.use_img = use_img
        self.pre_image_norm = pre_image_norm
        self.derive_next_image = bool(derive_next_image)
        self.derive_next_obs = bool(derive_next_obs)
        self.derive_next_action = bool(derive_next_action)
        if image_keys is None:
            self.image_key_map = {'image': 'img'}
        else:
            self.image_key_map = {key: key for key in image_keys}
        keys = [
            'state', 'action', 'point_cloud', 'reward', 'done', 'timeout', 'return'
        ]
        if not self.derive_next_obs:
            keys.extend(['next_state', 'next_point_cloud'])
        if not self.derive_next_action:
            keys.append('next_action')
        if self.use_img:
            for zarr_key in self.image_key_map.values():
                keys.append(zarr_key)
                if not self.derive_next_image:
                    keys.append(f'next_{zarr_key}')
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys)
        if self.derive_next_obs or self.derive_next_action or (
            self.use_img and self.derive_next_image
        ):
            n_frames = int(self.replay_buffer.episode_ends[-1])
            self.next_frame_index = np.minimum(
                np.arange(n_frames, dtype=np.int64) + 1,
                n_frames - 1,
            )
            self.next_frame_index[self.replay_buffer.episode_ends - 1] = (
                self.replay_buffer.episode_ends - 1
            )
            self.replay_buffer.root['data']['_frame_index'] = np.arange(
                n_frames, dtype=np.int64
            )
        # construct scaled reward and return
        # import pdb; pdb.set_trace()
        if scale_strategy == 'dynamic':
            print('scaling reward dynamically')
            reward_norm = RewardScaling(1, gamma=0.99)
            rewards = self.replay_buffer['reward'].flatten()
            for i, not_done in enumerate(1 - self.replay_buffer['done'].flatten()):
                if not not_done:
                    reward_norm.reset()
                else:
                    rewards[i] = reward_norm(rewards[i])
            self.replay_buffer.root['data']['reward'] = rewards.reshape(-1, 1)
            self.replay_buffer.root['data']['return'] = compute_return(self.replay_buffer['reward'], 1 - self.replay_buffer['done'], gamma=0.99)
            self.reward_norm = reward_norm
            cprint('reward and return scaled', 'green')
        # self.replay_buffer.root {'meta', 'data'}

        # for key, value in self.replay_buffer.items():
        #     cprint(f'Replay Buffer: {key}, shape {value.shape}, dtype {value.dtype}, range {value.min():.2f}~{value.max():.2f}', 'green')
        # cprint("--------------------------", 'green')

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            sequence_stride=sequence_stride)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.sequence_stride = sequence_stride
    def reward_scaling(self, scaling_strategy = 'dynamic', gamma = 0.99):
        if scaling_strategy == 'dynamic':
            print('scaling reward dynamically')
            reward_norm = RewardScaling(1, gamma)
            rewards = self.replay_buffer['reward'].flatten()
            for i, not_done in enumerate(1 - self.replay_buffer['done'].flatten()):
                if not not_done:
                    reward_norm.reset()
                else:
                    rewards[i] = reward_norm(rewards[i])
            self.replay_buffer['reward'] = rewards.reshape(-1, 1)
            self.replay_buffer['return'] = compute_return(self.replay_buffer['reward'], 1 - self.replay_buffer['done'], gamma)
        

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            sequence_stride=self.sequence_stride,
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        next_action = (
            self.replay_buffer['action']
            if self.derive_next_action
            else self.replay_buffer['next_action']
        )
        next_state = (
            self.replay_buffer['state']
            if self.derive_next_obs
            else self.replay_buffer['next_state']
        )
        next_point_cloud = (
            self.replay_buffer['point_cloud']
            if self.derive_next_obs
            else self.replay_buffer['next_point_cloud']
        )
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],

            'next_action': next_action,
            'next_agent_pos': next_state[...,:],
            'next_point_cloud': next_point_cloud,

            # 'reward': self.replay_buffer['reward'],
            # 'not_done': 1. - self.replay_buffer['done'],
            # 'return': self.replay_buffer['return'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        if self.use_img and self.pre_image_norm:
            for obs_key in self.image_key_map:
                normalizer[obs_key] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    @staticmethod
    def _prepare_image(image):
        image = np.asarray(image)
        is_integer = np.issubdtype(image.dtype, np.integer)
        image = image.astype(np.float32)
        if is_integer:
            image /= 255.0
        if image.ndim == 4 and image.shape[-1] == 3:
            image = np.moveaxis(image, -1, 1)
        return image

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        needs_next_indices = self.derive_next_obs or self.derive_next_action or (
            self.use_img and self.derive_next_image
        )
        next_indices = (
            self.next_frame_index[sample['_frame_index']]
            if needs_next_indices
            else None
        )
        if self.derive_next_obs:
            next_agent_pos = self.replay_buffer['state'][next_indices].astype(np.float32)
            next_point_cloud = self.replay_buffer['point_cloud'][next_indices].astype(np.float32)
        else:
            next_agent_pos = sample['next_state'][:,].astype(np.float32)
            next_point_cloud = sample['next_point_cloud'][:,].astype(np.float32)
        next_action = (
            self.replay_buffer['action'][next_indices].astype(np.float32)
            if self.derive_next_action
            else sample['next_action'].astype(np.float32)
        )

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
                # 'image': image, # T, 84, 84, 3
            },
            'next_obs': {
                'point_cloud': next_point_cloud, # T, 1024, 6
                'agent_pos': next_agent_pos, # T, D_pos
                # 'image': next_image, # T, 84, 84, 3
            }, 
            'reward': sample['reward'].astype(np.float32), # T, D_action
            'not_done': 1. - sample['done'].astype(np.bool_), # T, D_action
            'return': sample['return'].astype(np.float32), # T, D_action
            'action': sample['action'].astype(np.float32), # T, D_action
            'next_action': next_action # T, D_action
        }
        if self.use_img:
            for obs_key, zarr_key in self.image_key_map.items():
                data['obs'][obs_key] = self._prepare_image(sample[zarr_key])
                if self.derive_next_image:
                    next_image = self.replay_buffer[zarr_key][next_indices]
                else:
                    next_image = sample[f'next_{zarr_key}']
                data['next_obs'][obs_key] = self._prepare_image(next_image)

        return data
    def get_shape_info(self, n_action_steps, n_obs_steps):
        sample = self.sampler.sample_sequence(0)
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        # import pdb; pdb.set_trace()
        shape_info = {
        'obs': {
            'point_cloud': (n_obs_steps,) + point_cloud.shape[1:],
            'agent_pos': (n_obs_steps,) + agent_pos.shape[1:],
        },
        'action': (n_action_steps, sample['action'].shape[-1]),
        }
        if self.use_img:
            for obs_key, zarr_key in self.image_key_map.items():
                image = self._prepare_image(sample[zarr_key])
                shape_info['obs'][obs_key] = (n_obs_steps,) + image.shape[1:]
        else:
            shape_info['obs']['image'] = (n_obs_steps, 3, 84, 84)
        return shape_info
    def get_all_data(self,) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(range(self.replay_buffer['action'].shape[0]))
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_length(self, ):
        return len(self.sampler.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
