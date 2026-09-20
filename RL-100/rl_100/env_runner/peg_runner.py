from datetime import datetime
import os
import time

import numpy as np
import torch
import tqdm
from termcolor import cprint

from rl_100.common import logger_util
from rl_100.env_runner.base_runner import BaseRunner
from rl_100.gym_util.multistep_wrapper_real import MultiStepWrapper
from rl_100.policy.base_policy import BasePolicy


class PegRunner(BaseRunner):
    """Real-robot runner for the single-stage peg insertion task."""

    def __init__(
        self,
        output_dir,
        eval_episodes=20,
        max_steps=500,
        n_obs_steps=8,
        n_action_steps=8,
        fps=10,
        crf=22,
        render_size=84,
        tqdm_interval_sec=5.0,
        task_name='peg',
        use_point_crop=True,
        env_num=1,
        with_pointcloud=True,
        fake_env=False,
        num_points=1024,
        state_shape=7,
        dt=1 / 20,
        use_ee=False,
        gamma=0.99,
        robot_ip='192.168.1.202',
        smooth_penalty=0.001,
        success_reward=2.0,
        failure_reward=-1.0,
        require_reset_confirmation=True,
    ):
        super().__init__(output_dir)
        self.eval_episodes = int(eval_episodes)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.task_name = task_name
        self.tqdm_interval_sec = float(tqdm_interval_sec)
        self.fake_env = bool(fake_env)
        self.use_ee = bool(use_ee)
        self.handles_keyboard_result = True
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

        # These arguments are part of the common runner config interface.
        _ = (fps, crf, render_size, use_point_crop, env_num, with_pointcloud, state_shape)

        self.env = None
        if not self.fake_env:
            # Keep hardware imports lazy so offline jobs do not start listeners.
            from rl_100.env import PegEnv

            peg_env = PegEnv(
                robot_ip=robot_ip,
                dt=dt,
                num_point_cloud=num_points,
                max_episode_steps=max_steps,
                smooth_penalty=smooth_penalty,
                success_reward=success_reward,
                failure_reward=failure_reward,
                require_reset_confirmation=require_reset_confirmation,
            )
            self.env = MultiStepWrapper(
                peg_env,
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.n_action_steps,
                max_episode_steps=None,
                reward_agg_method='discounted_sum',
                gamma=gamma,
            )

    def run(
        self,
        policy: BasePolicy,
        data_collect=False,
        use_cm=False,
        distill2mean=False,
        traj_path=None,
    ):
        deterministic = not data_collect

        def predict(obs):
            return policy.predict_action(
                obs,
                deterministic=deterministic,
                use_cm=use_cm,
                distill2mean=distill2mean,
            )

        return self._run_rollouts(policy, predict, data_collect, traj_path)

    def idql_run(
        self,
        policy: BasePolicy,
        dynamics,
        first_action,
        get_np,
        use_gae,
        iql,
        Q,
        repeat_num,
        traj_path=None,
        data_collect=False,
        use_cm=False,
        distill2mean=False,
    ):
        def predict(obs):
            return policy.sample_action(
                obs,
                dynamics=dynamics,
                first_action=first_action,
                get_np=get_np,
                use_gae=use_gae,
                iql=iql,
                Q=Q,
                repeat_num=repeat_num,
                use_cm=use_cm,
                distill2mean=distill2mean,
            )

        return self._run_rollouts(policy, predict, data_collect, traj_path)

    def _run_rollouts(self, policy, predict, data_collect, traj_path):
        if self.fake_env:
            return self._build_log(success_rate=0.0, mean_return=0.0)
        if data_collect and traj_path is None:
            raise ValueError('traj_path is required when data_collect=True')
        if data_collect:
            os.makedirs(traj_path, exist_ok=True)

        episode_returns = []
        successes = 0
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        progress = tqdm.tqdm(
            range(self.eval_episodes),
            desc=f'Eval in Peg {self.task_name} Pointcloud Env',
            leave=False,
            mininterval=self.tqdm_interval_sec,
        )

        for episode_idx in progress:
            policy_obs = self.env.reset()
            policy.reset()
            episode_data = self._empty_episode_data() if data_collect else None
            episode_return = 0.0
            episode_success = False
            step_count = 0
            start_time = time.time()

            while True:
                policy_input = self._make_policy_input(policy_obs, policy.device)
                with torch.inference_mode():
                    action_result = predict(policy_input)
                actions = self._as_action_chunk(action_result)

                next_obs, rewards, dones, infos = self.env.step(actions.copy())
                rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
                dones = np.asarray(dones, dtype=bool).reshape(-1)
                executed = len(rewards)
                actions = actions[:executed]

                success = self._info_array(infos, 'is_success', executed)
                timeout = self._info_array(infos, 'timeout', executed)
                episode_return += float(rewards.sum())
                episode_success = episode_success or bool(success.any())
                step_count += executed

                if data_collect:
                    self._append_transitions(
                        episode_data,
                        policy_obs,
                        next_obs,
                        actions,
                        rewards,
                        dones,
                        timeout,
                        success,
                    )

                policy_obs = self._advance_observation_history(policy_obs, next_obs)
                if dones.any():
                    break

            episode_returns.append(episode_return)
            successes += int(episode_success)
            frequency = step_count / max(time.time() - start_time, 1e-6)
            cprint(
                f'Episode {episode_idx}: success={episode_success}, '
                f'return={episode_return:.3f}, action frequency={frequency:.2f} Hz',
                'green' if episode_success else 'yellow',
            )

            if data_collect:
                filename = os.path.join(
                    traj_path,
                    f'eval_episode_{timestamp}_{episode_idx}.h5',
                )
                self._save_episode(filename, episode_data)

        success_rate = successes / self.eval_episodes if self.eval_episodes else 0.0
        mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
        return self._build_log(success_rate, mean_return)

    def _make_policy_input(self, obs, device):
        state_key = 'ee_pose' if self.use_ee else 'agent_pos'
        return {
            'point_cloud': torch.as_tensor(
                obs['point_cloud'][-self.n_obs_steps:], device=device, dtype=torch.float32
            ).unsqueeze(0),
            'agent_pos': torch.as_tensor(
                obs[state_key][-self.n_obs_steps:], device=device, dtype=torch.float32
            ).unsqueeze(0),
            'image': torch.as_tensor(
                obs['image'][-self.n_obs_steps:], device=device, dtype=torch.float32
            ).unsqueeze(0),
        }

    def _as_action_chunk(self, action_result):
        if not isinstance(action_result, dict) or 'action' not in action_result:
            raise TypeError("policy output must be a dict containing 'action'")
        action = action_result['action']
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action)
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]
        if action.ndim == 1:
            action = action[None]
        if action.ndim != 2 or action.shape[-1] != 8:
            raise ValueError(f'policy action must have shape (T, 8), got {action.shape}')
        if len(action) == 0:
            raise ValueError('policy returned an empty action chunk')
        return action[:self.n_action_steps]

    def _advance_observation_history(self, old_obs, new_obs):
        return {
            key: np.concatenate([old_obs[key], new_obs[key]], axis=0)[-self.n_obs_steps:]
            for key in old_obs
        }

    @staticmethod
    def _info_array(infos, key, length):
        values = infos.get(key, [False] * length)
        values = np.asarray(values, dtype=bool).reshape(-1)
        if len(values) < length:
            values = np.pad(values, (0, length - len(values)), constant_values=False)
        return values[:length]

    @staticmethod
    def _empty_episode_data():
        keys = (
            'state', 'action', 'point_cloud',
            'next_state', 'next_point_cloud',
            'reward', 'done', 'timeout', 'is_success',
        )
        return {key: [] for key in keys}

    def _append_transitions(
        self,
        data,
        current_obs,
        next_obs,
        actions,
        rewards,
        dones,
        timeout,
        success,
    ):
        state_key = 'ee_pose' if self.use_ee else 'agent_pos'
        for idx, action in enumerate(actions):
            previous_idx = idx - 1
            data['state'].append(
                current_obs[state_key][-1] if idx == 0 else next_obs[state_key][previous_idx]
            )
            data['point_cloud'].append(
                current_obs['point_cloud'][-1] if idx == 0 else next_obs['point_cloud'][previous_idx]
            )
            data['action'].append(action)
            data['next_state'].append(next_obs[state_key][idx])
            data['next_point_cloud'].append(next_obs['point_cloud'][idx])
            data['reward'].append(rewards[idx])
            data['done'].append(dones[idx])
            data['timeout'].append(timeout[idx])
            data['is_success'].append(success[idx])

    @staticmethod
    def _save_episode(filename, data):
        import h5py

        actions = np.asarray(data['action'])
        next_action = np.concatenate([actions[1:], actions[-1:]], axis=0)
        with h5py.File(filename, 'w') as hdf5_file:
            for key, values in data.items():
                hdf5_file.create_dataset(key, data=np.asarray(values))
            hdf5_file.create_dataset('next_action', data=next_action)
        cprint(f'Data saved in {filename}', 'green')

    def _build_log(self, success_rate, mean_return):
        self.logger_util_test.record(success_rate)
        self.logger_util_test10.record(success_rate)
        log_data = {
            'mean_success_rates': success_rate,
            'test_mean_score': success_rate,
            'mean_returns': mean_return,
            'SR_test_L3': self.logger_util_test.average_of_largest_K(),
            'SR_test_L5': self.logger_util_test10.average_of_largest_K(),
        }
        cprint(f'test_mean_score: {success_rate}', 'green')
        cprint(f'mean_returns: {mean_return}', 'green')
        return log_data
