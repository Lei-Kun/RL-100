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
            sequence_stride=1,
            derive_next_action=False,
            ):
        super().__init__()
        self.task_name = task_name
        self.use_img = use_img
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud', 'next_state', 'next_action', 'next_point_cloud', 'reward', 'done', 'timeout', 'return'])
        if derive_next_action:
            action = self.replay_buffer['action']
            next_action = action.copy()
            episode_start = 0
            for episode_end in self.replay_buffer.episode_ends:
                next_action[episode_start:episode_end - 1] = action[episode_start + 1:episode_end]
                next_action[episode_end - 1] = action[episode_end - 1]
                episode_start = episode_end
            self.replay_buffer.root['data']['next_action'] = next_action
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
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],

            'next_action': self.replay_buffer['next_action'],
            'next_agent_pos': self.replay_buffer['next_state'][...,:],
            'next_point_cloud': self.replay_buffer['next_point_cloud'],

            # 'reward': self.replay_buffer['reward'],
            # 'not_done': 1. - self.replay_buffer['done'],
            # 'return': self.replay_buffer['return'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        # image = sample['img'][:,].astype(np.float32) # (T, 3, 64, 64)

        # image = np.random.rand(point_cloud.shape[0], 3, 84, 84)  # dummy image
        
        next_agent_pos = sample['next_state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        next_point_cloud = sample['next_point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        # next_image = np.random.rand(point_cloud.shape[0], 3, 84, 84)  # dummy image

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
            'next_action': sample['next_action'].astype(np.float32) # T, D_action
        }

        return data
    def get_shape_info(self, n_action_steps, n_obs_steps):
        sample = self.sampler.sample_sequence(10)
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        # import pdb; pdb.set_trace()
        image = np.random.rand(point_cloud.shape[0], 3, 84, 84)  # dummy image
        # image = sample['img'][:,].astype(np.float32) # (T, 3, 64, 64)

        shape_info = {
        'obs': {
            'point_cloud': (n_obs_steps,) + point_cloud.shape[1:],
            'agent_pos': (n_obs_steps,) + agent_pos.shape[1:],
            'image': (n_obs_steps,) + image.shape[1:],
        },
        'action': (n_action_steps, sample['action'].shape[-1]),
        }
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
