import wandb
import numpy as np
import torch
import tqdm
from diffusion_policy_3d.gym_util.multistep_wrapper_multiobs import MultiStepWrapper, aggregate, dict_take_last_n
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util

from termcolor import cprint
import time
import copy
import torch
import numpy as np
import h5py
import tqdm
import os
from copy import deepcopy
from diffusion_policy_3d.env import JuicingEnv


class JuicingRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 env_num=1,
                 with_pointcloud=True,
                 fake_env=False,
                 num_points=1024,
                 state_shape=7,  # 6 DOF pose + 1 gripper state
                 dt=1/30,
                 use_ee=False,
                 episode_log_enabled=False,
                 episode_log_dir=None,
                 stop_mode='auto',
                 enable_safety_guard=True,
                 max_pos_step_mm=12.0,
                 max_rot_step_deg=12.0,
                 fault_cmd_pos_mm=50.0,
                 fault_cmd_rot_deg=45.0,
                 max_measured_pos_jump_m=0.003,
                 fault_measured_rot_deg=15.0,
                 ):
        super().__init__(output_dir)
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        self.fake_env = fake_env
        self.eval_episodes = eval_episodes
        self.task_name = task_name
        self.state_shape = state_shape
        self.dt = dt
        if self.fake_env:
            self.env = None
        else:
            self.env = MultiStepWrapper(
                JuicingEnv(
                    dt=dt,
                    num_point_cloud=num_points,
                    episode_log_enabled=episode_log_enabled,
                    episode_log_dir=episode_log_dir,
                    episode_log_prefix=task_name or 'juicing',
                    stop_mode=stop_mode,
                    enable_safety_guard=enable_safety_guard,
                    max_pos_step_mm=max_pos_step_mm,
                    max_rot_step_deg=max_rot_step_deg,
                    fault_cmd_pos_mm=fault_cmd_pos_mm,
                    fault_cmd_rot_deg=fault_cmd_rot_deg,
                    max_measured_pos_jump_m=max_measured_pos_jump_m,
                    fault_measured_rot_deg=fault_measured_rot_deg,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.use_ee = use_ee
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)


    def run(self, policy: BasePolicy, data_collect = False, use_cm = False, distill2mean=False, traj_path = None): 
        if data_collect:
            deterministic = False
        else:
            deterministic = True
        if self. fake_env:
            all_goal_achieved = np.random.randint(90, 100)
            all_success_rates = np.random.randint(90, 100)
            all_returns = 100
            log_data = {}
            log_data['mean_n_goal_achieved'] = np.mean(all_success_rates)
            log_data['mean_success_rates'] = all_success_rates

            log_data['test_mean_score'] = all_success_rates
            log_data['mean_returns'] = np.mean(all_returns)
            cprint(f"test_mean_score: {all_success_rates}", 'green')
            cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
            self.logger_util_test.record(all_success_rates)
            self.logger_util_test10.record(all_success_rates)
            log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
            log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
            return log_data
        else:
            device = policy.device
            dtype = policy.dtype
            env = self.env
            base_env = getattr(env, 'env', env)
            if hasattr(base_env, 'set_episode_log_context'):
                base_env.set_episode_log_context(task_name=self.task_name, stage_label='rollout')

            all_goal_achieved = []
            all_success_rates = []
            all_returns = []
            hard_success = 0
            from datetime import datetime

            file_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            completed_episodes = 0
            pbar = tqdm.tqdm(total=self.eval_episodes,
                             desc=f"Eval in Juicing {self.task_name} Pointcloud Env",
                             leave=False, mininterval=self.tqdm_interval_sec)

            while completed_episodes < self.eval_episodes:
                # 每个 episode 初始化数据存储列表
                point_cloud_arrays = []
                state_arrays = []
                action_arrays = []
                depth_arrays = []
                depth_scale_arrays = []
                
                next_point_cloud_arrays = []
                next_state_arrays = []   
                next_action_arrays = []
                next_depth_arrays = []
                next_depth_scale_arrays = []
                
                reward_arrays = []
                done_arrays = []
                timeout_arrays = []
                is_success_arrays = []
                
                # 开始 rollout
                obs = env.reset()
                policy.reset()
                
                episode_done = False 
                num_goal_achieved = 0
                actual_step_count = 0
                episode_reward  = 0
                total_count = 0
                time_start = time.time()
                is_success = False
                pre_reward = -1

                while not episode_done:
                    # 保存当前状态信息
                    time_frame = time.time()
                    # if self.use_ee:
                    #     obs['agent_pos'] = obs['ee_pose']
                    obs_dict_input = {}
                    
                    # Debug: print shapes
                    # print(f"obs['agent_pos'] shape: {obs['agent_pos'].shape if hasattr(obs['agent_pos'], 'shape') else len(obs['agent_pos'])}")
                    # print(f"n_obs_steps: {self.n_obs_steps}")
                    
                    obs_dict_input['agent_pos'] = np.array(obs['agent_pos'][-self.n_obs_steps:])
                    # print(f"obs_dict_input['agent_pos'] shape after slicing: {obs_dict_input['agent_pos'].shape}")
                    
                    obs_dict_input['point_cloud'] = np.array(obs['point_cloud'][-self.n_obs_steps:])
                    obs_dict_input['depth'] = np.array(obs['depth'][-self.n_obs_steps:])
                    obs_dict_input['depth_scale'] = np.array(obs['depth_scale'][-self.n_obs_steps:])
                    # obs_dict_input['image'] = np.array(obs['image'][-self.n_obs_steps:])
                    obs_dict_input['ee_pose'] = np.array(obs['ee_pose'][-self.n_obs_steps:])

                    state_arrays.extend(obs['agent_pos'])
                    point_cloud_arrays.extend(obs['point_cloud'])
                    depth_arrays.extend(obs['depth'])
                    depth_scale_arrays.extend(obs['depth_scale'])
                    
                    # 构造策略输入字典（增加 batch 维度）
                    infer_time = time.time()
                    
                    # 将 numpy 转换为 torch tensor 并移动到指定设备
                    obs_dict = dict_apply(obs_dict_input,
                                        lambda x: torch.from_numpy(x).to(device=device))
                    
                    with torch.no_grad():
                        obs_dict_input_torch = {
                            'point_cloud': obs_dict['point_cloud'].unsqueeze(0).to(torch.float)
                        }
                        obs_dict_input_torch['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                        # obs_dict_input_torch['image'] = obs_dict['image'].unsqueeze(0).to(torch.float)
                        obs_dict_input_torch['depth'] = obs_dict['depth'].unsqueeze(0).to(torch.float)
                        obs_dict_input_torch['depth_scale'] = obs_dict['depth_scale'].unsqueeze(0).to(torch.float)
                        obs_dict_input_torch['ee_pose'] = obs_dict['ee_pose'].unsqueeze(0).to(torch.float)
                        action_dict = policy.predict_action(obs_dict_input_torch, deterministic=deterministic, use_cm=use_cm, distill2mean=distill2mean)
                        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                        action = np_action_dict['action'].squeeze(0)
                    # print(f"action shape from policy: {action.shape}")
                    # print(f"n_action_steps configured: {self.n_action_steps}")
                    action_arrays.extend(action)
                    # import pdb; pdb.set_trace()  # Removed debug breakpoint
                    obs, reward, done, info = env.step_offline(action.copy())
                    next_state_arrays.extend(obs['agent_pos'])
                    next_point_cloud_arrays.extend(obs['point_cloud'])
                    next_depth_arrays.extend(obs['depth'])
                    next_depth_scale_arrays.extend(obs['depth_scale'])

                    
                    if isinstance(done, list):
                        done_arrays.extend(done)
                    else:
                        done_arrays.append(done)
                    timeout_arrays.extend(info['timeout'])
                    is_success_arrays.extend(info['is_success'])
                    if reward is None:
                        reward = pre_reward
                    # print('reward:', reward)
                    # Flatten reward if it's a nested list
                    if isinstance(reward, float) or isinstance(reward, int):
                        reward = [reward]
                    if isinstance(reward[0], list):
                        reward = [r for sublist in reward for r in sublist]
                    episode_reward += sum(reward)
                    reward_arrays.extend(reward)
                    pre_reward = reward
                        
                    actual_step_count += 1
                    episode_done = np.array(done).any()
                    print('Step freq:', 1 / (time.time() - time_frame))
                
                
                # 保存最后一步的next depth
                final_depth = obs['depth'][-1]
                final_depth_scale = obs['depth_scale'][-1]
                
                # Debug: Print array lengths before adjustment
                # print(f"Before adjustment:")
                # print(f"  state_arrays: {len(state_arrays)}")
                # print(f"  point_cloud_arrays: {len(point_cloud_arrays)}")
                # print(f"  depth_arrays: {len(depth_arrays)}")
                # print(f"  depth_scale_arrays: {len(depth_scale_arrays)}")
                # print(f"  next_state_arrays: {len(next_state_arrays)}")
                # print(f"  next_point_cloud_arrays: {len(next_point_cloud_arrays)}")
                # print(f"  action_arrays: {len(action_arrays)}")
                
                action_arrays = action_arrays[:len(next_point_cloud_arrays)]
                extra_step = len(next_point_cloud_arrays) - len(point_cloud_arrays)
                
                point_cloud_arrays.extend(next_point_cloud_arrays[-extra_step-1:-1])
                depth_arrays.extend(next_depth_arrays[-extra_step-1:-1])
                depth_scale_arrays.extend(next_depth_scale_arrays[-extra_step-1:-1])
                state_arrays.extend(next_state_arrays[-extra_step-1:-1])
                # 结束回合时，再对最后一步做一次动作预测（保证 next_action_arrays 对齐）
                obs_dict = dict_apply(obs,
                                    lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    obs_dict_input = {
                        'point_cloud': obs_dict['point_cloud'].unsqueeze(0).to(torch.float)
                    }
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                    # obs_dict_input['image'] = obs_dict['image'].unsqueeze(0).to(torch.float)
                    action_dict = policy.predict_action(obs_dict_input, deterministic=True, use_cm=use_cm, distill2mean=distill2mean)
                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action'].squeeze(0)
                

                # 处理 next_action_arrays，使用 deepcopy 避免引用问题，然后调整索引
                next_action_arrays = copy.deepcopy(action_arrays)
                next_action_arrays.append(action[0])
                next_action_arrays = next_action_arrays[1:]
                
                time_end = time.time()
                print('action frequency: ', actual_step_count / (time_end - time_start))
                all_returns.append(episode_reward)

                # Ask user for success/failure after episode ends
                print("\nEpisode ended. Get flag from info")
                # user_input = input("Type number for success, letter for failure: ").strip()
                
                # Use the last character to avoid interference from stop key
                # if user_input:
                #     user_input = user_input[-1]
                # if user_input and user_input[0].isdigit():
                #     # Record as success
                #     if is_success_arrays:
                #         is_success_arrays[-1] = True
                #     if not is_success:
                #         is_success = True
                #         hard_success += 1
                if info['is_success'][-1]:
                    if is_success_arrays:
                        is_success_arrays[-1] = True
                    hard_success += 1
                    print(f"Episode {completed_episodes}: Success recorded!")
                else:
                    # Record as failure  
                    print(f"Episode {completed_episodes}: Failure recorded!")
                
                # time.sleep(3)
                
                print("success rate: ", hard_success / (completed_episodes+1))
                print("is success: ", info['is_success'][-1])

                env.reset_end()

                # Save data to HDF5 if data collection is enabled
                if data_collect:
                    hdf5_filename = os.path.join(traj_path, 'eval_episode_{}_{}.h5'.format(str(file_time), completed_episodes))
                    with h5py.File(hdf5_filename, 'w') as hdf5_file:
                        hdf5_file.create_dataset('point_cloud', data=np.array(point_cloud_arrays))
                        hdf5_file.create_dataset('depth', data=np.array(depth_arrays))
                        hdf5_file.create_dataset('depth_scale', data=np.array(depth_scale_arrays))
                        hdf5_file.create_dataset('state', data=np.array(state_arrays))
                        hdf5_file.create_dataset('action', data=np.array(action_arrays))
                        hdf5_file.create_dataset('next_point_cloud', data=np.array(next_point_cloud_arrays))
                        # 保存最后的depth到一个单独的数据集
                        hdf5_file.create_dataset('final_depth', data=final_depth)
                        hdf5_file.create_dataset('final_depth_scale', data=final_depth_scale)
                        # 不需要保存next_depth_arrays，因为它们就是下一步的depth_arrays
                        hdf5_file.create_dataset('next_state', data=np.array(next_state_arrays))
                        hdf5_file.create_dataset('next_action', data=np.array(next_action_arrays))
                        hdf5_file.create_dataset('reward', data=np.array(reward_arrays))
                        hdf5_file.create_dataset('done', data=np.array(done_arrays))
                        hdf5_file.create_dataset('timeout', data=np.array(timeout_arrays))
                        hdf5_file.create_dataset('is_success', data=np.array(is_success_arrays))
                        print('Data saved in {}'.format(hdf5_filename))

                completed_episodes += 1
                pbar.update(1)

            pbar.close()

            # log
            log_data = dict()

            all_success_rates = hard_success / self.eval_episodes

            log_data['mean_success_rates'] = all_success_rates

            log_data['test_mean_score'] = all_success_rates
            log_data['mean_returns'] = np.mean(all_returns)
            cprint(f"test_mean_score: {all_success_rates}", 'green')
            cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
            self.logger_util_test.record(all_success_rates)
            self.logger_util_test10.record(all_success_rates)
            log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
            log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

            del env

            return log_data
    
    def idql_run(self, policy: BasePolicy, dynamics, first_action, get_np, use_gae, iql, Q, repeat_num, traj_path = None, data_collect = None, use_cm=False, distill2mean=False):
        if data_collect:
            deterministic = False
        else:
            deterministic = True

        if self.fake_env:
            all_goal_achieved = np.random.randint(90, 100)
            all_success_rates = np.random.randint(90, 100)
            all_returns = 100
            log_data = {}
            log_data['mean_n_goal_achieved'] = np.mean(all_success_rates)
            log_data['mean_success_rates'] = all_success_rates

            log_data['test_mean_score'] = all_success_rates
            log_data['mean_returns'] = np.mean(all_returns)
            cprint(f"test_mean_score: {all_success_rates}", 'green')
            cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
            self.logger_util_test.record(all_success_rates)
            self.logger_util_test10.record(all_success_rates)
            log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
            log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
            return log_data
        else:
            device = policy.device
            dtype = policy.dtype
            env = self.env

            all_goal_achieved = []
            all_success_rates = []
            all_returns = []
            hard_success = 0
            from datetime import datetime

            file_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # 使用 'a' 模式打开，方便追加写入数据
            for episode_idx in tqdm.tqdm(range(self.eval_episodes),
                                        desc=f"Eval in Juicing {self.task_name} Pointcloud Env with IDQL",
                                        leave=False, mininterval=self.tqdm_interval_sec):
                
                # 每个 episode 初始化数据存储列表
                point_cloud_arrays = []
                state_arrays = []
                action_arrays = []
                depth_arrays = []
                depth_scale_arrays = []
                
                next_point_cloud_arrays = []
                next_state_arrays = []   
                next_action_arrays = []
                
                reward_arrays = []
                done_arrays = []
                timeout_arrays = []
                is_success_arrays = []
                
                # 开始 rollout
                obs = env.reset()
                policy.reset()
                
                done = False 
                num_goal_achieved = 0
                actual_step_count = 0
                episode_reward  = 0
                total_count = 0
                time_start = time.time()
                is_success = False
                pre_reward = -1
                
                while not done:
                    # 保存当前状态信息
                    time_action = time.time()
                    np_obs_dict = obs
                    
                    state_arrays.append(obs['agent_pos'][-1])
                    point_cloud_arrays.append(obs['point_cloud'][-1])
                    depth_arrays.append(obs['depth'][-1])
                    depth_scale_arrays.append(obs['depth_scale'][-1])
                    
                    # 将 numpy 转换为 torch tensor 并移动到指定设备
                    obs_dict = dict_apply(np_obs_dict,
                                        lambda x: torch.from_numpy(x).to(device=device))
                    
                    # 构造策略输入字典（增加 batch 维度）
                    with torch.no_grad():
                        obs_dict_input = {
                            'point_cloud': obs_dict['point_cloud'].unsqueeze(0).to(torch.float)
                        }
                        obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                        # obs_dict_input['image'] = obs_dict['image'].unsqueeze(0).to(torch.float)
                        
                        for key in obs_dict_input.keys():
                            print('$$',key)
                            print(obs_dict_input[key].shape)
                        
                        # Use sample_action for IDQL instead of predict_action
                        action_dict = policy.sample_action(obs_dict_input, dynamics=dynamics, first_action=first_action, get_np=get_np, use_gae=use_gae, iql=iql, Q=Q, repeat_num=repeat_num,
                                                          use_cm=use_cm, distill2mean=distill2mean)
                    
                    # 将动作从 tensor 转换为 numpy 并保存
                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action'].squeeze(0)
                    action_arrays.append(action)
                    
                    # 执行环境一步
                    obs, reward, done, info = env.step_offline(action.copy())
                    
                    # Ensure done is a list (handle both list and scalar returns)
                    if not isinstance(done, list):
                        done = [done]
                    if not isinstance(reward, list):
                        reward = [reward]
                    
                    next_state_arrays.append(obs['agent_pos'][-1])
                    next_point_cloud_arrays.append(obs['point_cloud'][-1])
                    
                    done_arrays.append(done[0])
                    timeout_arrays.append(info['timeout'][-1])
                    is_success_arrays.append(bool(info['is_success'][-1]))
                    
                    if reward is None:
                        reward = pre_reward
                    episode_reward += sum(reward)
                    reward_arrays.append(reward[0])
                    pre_reward = reward
                    
                    # Don't use info['is_success'] since it's always False
                    # Success will be determined by user input after episode ends
                        
                    actual_step_count += 1
                    done = np.array(done).any()
                    print(1 / (time.time() - time_action))
                
                # 保存最后一步的next depth（这是唯一没有对应下一个current depth的）
                final_depth = obs['depth'][-1]
                final_depth_scale = obs['depth_scale'][-1]
                
                # 结束回合时，再对最后一步做一次动作预测（保证 next_action_arrays 对齐）
                obs_dict = dict_apply(obs,
                                    lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    obs_dict_input = {
                        'point_cloud': obs_dict['point_cloud'].unsqueeze(0).to(torch.float)
                    }
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0).to(torch.float)
                    # obs_dict_input['image'] = obs_dict['image'].unsqueeze(0).to(torch.float)
                    action_dict = policy.predict_action(obs_dict_input, deterministic=True, use_cm=use_cm, distill2mean=distill2mean)
                    np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action'].squeeze(0)
                
                # 处理 next_action_arrays，使用 deepcopy 避免引用问题，然后调整索引
                next_action_arrays = copy.deepcopy(action_arrays)
                next_action_arrays.append(action)
                next_action_arrays = next_action_arrays[1:]
                
                time_end = time.time()
                print('action frequency: ', actual_step_count / (time_end - time_start))
                all_returns.append(episode_reward)
                time.sleep(4)
                
                # 将本 episode 的数据保存到 HDF5 文件中
                if traj_path is not None:
                    hdf5_filename = os.path.join(traj_path, 'eval_episode_{}_{}.h5'.format(str(file_time), episode_idx))
                    with h5py.File(hdf5_filename, 'w') as hdf5_file:
                        hdf5_file.create_dataset('point_cloud', data=np.array(point_cloud_arrays))
                        hdf5_file.create_dataset('depth', data=np.array(depth_arrays))
                        hdf5_file.create_dataset('depth_scale', data=np.array(depth_scale_arrays))
                        
                        hdf5_file.create_dataset('state', data=np.array(state_arrays))
                        
                        hdf5_file.create_dataset('action', data=np.array(action_arrays))
                        hdf5_file.create_dataset('next_point_cloud', data=np.array(next_point_cloud_arrays))
                        # 保存最后的depth到一个单独的数据集
                        hdf5_file.create_dataset('final_depth', data=final_depth)
                        hdf5_file.create_dataset('final_depth_scale', data=final_depth_scale)
                        # 不需要保存next_depth_arrays，因为它们就是下一步的depth_arrays
                        hdf5_file.create_dataset('next_state', data=np.array(next_state_arrays))
                        hdf5_file.create_dataset('next_action', data=np.array(next_action_arrays))
                        hdf5_file.create_dataset('reward', data=np.array(reward_arrays))
                        hdf5_file.create_dataset('done', data=np.array(done_arrays, dtype=bool))
                        hdf5_file.create_dataset('timeout', data=np.array(timeout_arrays, dtype=bool))
                        hdf5_file.create_dataset('is_success', data=np.array(is_success_arrays, dtype=bool))
                        print('data save in {}'.format(hdf5_filename))
                
                print("success rate: ", hard_success / (episode_idx+1))
                print("is success: ", is_success)
                
            # log
            log_data = dict()

            all_success_rates = hard_success / self.eval_episodes

            log_data['mean_success_rates'] = all_success_rates

            log_data['test_mean_score'] = all_success_rates
            log_data['mean_returns'] = np.mean(all_returns)
            cprint(f"test_mean_score: {all_success_rates}", 'green')
            cprint(f"mean_returns: {np.mean(all_returns)}", 'green')
            self.logger_util_test.record(all_success_rates)
            self.logger_util_test10.record(all_success_rates)
            log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
            log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

            del env

            return log_data
