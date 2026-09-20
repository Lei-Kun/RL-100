import gym
from gym import spaces
import numpy as np
import torch
from collections import defaultdict, deque
import dill


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x[-n:])
    else:
        return np.array(x[-n:])



def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method='max', gamma=0.99):
    if isinstance(data[0], torch.Tensor):
        if method == 'max':
            # equivalent to any
            return torch.max(torch.stack(data))
        elif method == 'min':
            # equivalent to all
            return torch.min(torch.stack(data))
        elif method == 'mean':
            return torch.mean(torch.stack(data))
        elif method == 'sum':
            return torch.sum(torch.stack(data))
        elif method == 'discounted_sum':
            # 使用gamma折扣的求和
            data_tensor = torch.stack(data)
            gamma_weights = torch.pow(torch.tensor(gamma, dtype=data_tensor.dtype, device=data_tensor.device), 
                                    torch.arange(len(data), dtype=data_tensor.dtype, device=data_tensor.device))
            return torch.sum(data_tensor * gamma_weights)
        else:
            raise NotImplementedError()
    else:
        if method == 'max':
            # equivalent to any
            return np.max(data)
        elif method == 'min':
            # equivalent to all
            return np.min(data)
        elif method == 'mean':
            return np.mean(data)
        elif method == 'sum':
            return np.sum(data)
        elif method == 'discounted_sum':
            # 使用gamma折扣的求和
            data_array = np.array(data)
            gamma_weights = np.power(gamma, np.arange(len(data)))
            return np.sum(data_array * gamma_weights)
        else:
            raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    if isinstance(all_obs[0], np.ndarray):
        result = np.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], torch.Tensor):
        result = torch.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], dict):
        # Handle dictionary observations (e.g., multi-view images)
        result = {}
        for key in all_obs[0].keys():
            if isinstance(all_obs[0][key], np.ndarray):
                # Stack arrays for this key
                key_obs = [obs[key] for obs in all_obs]
                key_result = np.zeros((n_steps,) + key_obs[-1].shape, dtype=key_obs[-1].dtype)
                start_idx = -min(n_steps, len(key_obs))
                key_result[start_idx:] = np.array(key_obs[start_idx:])
                if n_steps > len(key_obs):
                    # pad
                    key_result[:start_idx] = key_result[start_idx]
                result[key] = key_result
            else:
                # For non-array values, just take the last value
                result[key] = all_obs[-1][key]
    else:
        raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='discounted_sum',
            gamma=0.99
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.gamma = gamma
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_action_steps+1))
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_action_steps+1))

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def step_online(self, action, gamma=None):
        """
        Online PPO path: preserve chunk-level transitions (stride=n_action_steps).
        Returns stacked latest observation, scalar aggregated reward, scalar done, and chunked info.
        When gamma is provided, updates wrapper gamma for discounted reward aggregation.

        actions: (n_action_steps,) + action_shape
        """
        rewards, dones = [], []
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, reward, done, info = super().step(act)
            # print('reward:', reward)

            self.obs.append(observation)
            rewards.append(reward)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            dones.append(done)
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        # 如果使用sum方法，改为使用discounted_sum
        # print('rewards:', rewards)
        if gamma is not None:
            self.gamma = gamma
        if self.reward_agg_method == 'sum':
            reward = aggregate(rewards, 'discounted_sum', self.gamma)
        else:
            reward = aggregate(rewards, self.reward_agg_method, self.gamma)
        print('agg reward:', reward)   
        done = aggregate(dones, 'max')
        info = dict_take_last_n(self.info, self.n_action_steps)
        
        # Special handling for goal_achieved: sum all occurrences during action chunk execution
        if 'goal_achieved' in self.info:
            # Get all goal_achieved values from the current action chunk execution
            all_goal_achieved = list(self.info['goal_achieved'])[-len(rewards):]
            info['goal_achieved'] = sum(all_goal_achieved)
            # import pdb; pdb.set_trace()

        return observation, reward, done, info

    def step(self, action):
        """
        Offline/data-collect path: preserve per-substep trajectories (stride=1).
        Returns per-substep observation arrays, per-substep rewards/dones/infos.

        actions: (n_action_steps,) + action_shape
        """
        observations, rewards, dones, infos = {}, [], [], {}
        for act in action:
            observation, reward, done, info = super().step(act)
            # print('reward:', reward)

            if len(observations.keys()) != 0:
                for key in observations.keys():
                    observations[key].append(observation[key])
            else:
                for key in observation.keys():
                    observations[key] = [observation[key]]
            rewards.append(reward)
            self.reward.append(reward)
            self.obs.append(observation)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            dones.append(done)
            self._add_info(info)
            if len(infos.keys()) != 0:
                for key in infos.keys():
                    infos[key].append(info[key])
            else:
                for key in info.keys():
                    infos[key] = [info[key]]
            if done:
                break

        # 如果使用sum方法，改为使用discounted_sum
        if self.reward_agg_method == 'sum':
            reward = aggregate(rewards, 'discounted_sum', self.gamma)
        else:
            reward = aggregate(rewards, self.reward_agg_method, self.gamma)
        print('agg reward:', reward)   
        done = aggregate(dones, 'max') 
        for key in observations.keys():
            # depth 和 depth_scale 保持原始数据类型，其他转成 float32
            if key in ['depth', 'depth_scale']:
                observations[key] = np.array(observations[key])
            else:
                observations[key] = np.array(observations[key], dtype=np.float32)
        return observations, rewards, dones, infos

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
