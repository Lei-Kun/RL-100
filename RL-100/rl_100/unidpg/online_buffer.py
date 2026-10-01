import torch
import numpy as np
from rl_100.common.pytorch_util import dict_apply

class ReplayBuffer:
    def __init__(self, args, shape_info,  device, wo_visual=False):
        self.use_imagin_robot = False
        for key in shape_info['obs']:
            if 'imagin_robot' in key:
                self.use_imagin_robot = True
                break
        self.wo_visual = wo_visual
        self.image_shapes = {}
        for key, shape in shape_info['obs'].items():
            shape = tuple(shape)
            if len(shape) == 4 and (shape[1] == 3 or shape[-1] == 3):
                if shape[-1] == 3 and shape[1] != 3:
                    shape = (shape[0], shape[-1], shape[1], shape[2])
                self.image_shapes[key] = shape
        self.image_keys = list(self.image_shapes)
        self.use_legacy_image = 'image' in self.image_shapes
        if not wo_visual:   
            self.point_cloud = np.zeros((args.batch_size, *shape_info['obs']['point_cloud']))
            if self.use_legacy_image:
                self.image = np.zeros((args.batch_size, *self.image_shapes['image']))
            else:
                self.images = {
                    key: np.zeros((args.batch_size, *shape), dtype=np.float32)
                    for key, shape in self.image_shapes.items()
                }
            if self.use_imagin_robot:
                self.imagin_robot = np.zeros((args.batch_size, *shape_info['obs']['imagin_robot']))
        self.agent_pos = np.zeros((args.batch_size, *shape_info['obs']['agent_pos']))
        self.action = np.zeros((args.batch_size, args.num_inference_steps + 1, *shape_info['action']))
        self.a_logprob = np.zeros((args.batch_size, args.num_inference_steps, *shape_info['action']))

        if not wo_visual:
            self.next_point_cloud = np.zeros((args.batch_size, *shape_info['obs']['point_cloud']))
            if self.use_legacy_image:
                self.next_image = np.zeros((args.batch_size, *self.image_shapes['image']))
            else:
                self.next_images = {
                    key: np.zeros((args.batch_size, *shape), dtype=np.float32)
                    for key, shape in self.image_shapes.items()
                }
            if self.use_imagin_robot:
                self.next_imagin_robot = np.zeros((args.batch_size, *shape_info['obs']['imagin_robot']))

        self.next_agent_pos = np.zeros((args.batch_size, *shape_info['obs']['agent_pos']))
        self.reward = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.dw = np.zeros((args.batch_size, 1))
        self.count = 0
        self.device = device

    def _get_image_array(self, key, next_obs=False):
        if key == 'image' and self.use_legacy_image:
            return self.next_image if next_obs else self.image
        return self.next_images[key] if next_obs else self.images[key]

    def store(self, obs, action, a_logprob, reward, next_obs, done, dw):
        if not self.wo_visual:
            self.point_cloud[self.count] = obs['point_cloud']
            for key in self.image_keys:
                self._get_image_array(key)[self.count] = obs[key]
            if self.use_imagin_robot:
                self.imagin_robot[self.count] = obs['imagin_robot']
        # import pdb; pdb.set_trace()
        self.agent_pos[self.count] = obs['agent_pos']
        self.action[self.count] = action
        self.a_logprob[self.count] = a_logprob
        self.reward[self.count] = reward
        if not self.wo_visual:
            self.next_point_cloud[self.count] = next_obs['point_cloud']
            for key in self.image_keys:
                self._get_image_array(key, next_obs=True)[self.count] = next_obs[key]
            if self.use_imagin_robot:
                self.next_imagin_robot[self.count] = next_obs['imagin_robot']
        self.next_agent_pos[self.count] = next_obs['agent_pos']
        self.done[self.count] = done
        self.dw[self.count] = dw
        self.count += 1

    def sample(
        self, batch_size: int
    ) -> tuple:

        ind = np.random.randint(0, int(self.count), size=batch_size)
        if not self.wo_visual:
            point_cloud = torch.FloatTensor(self.point_cloud[ind]).to(self.device)
            images = {
                key: torch.FloatTensor(self._get_image_array(key)[ind]).to(self.device)
                for key in self.image_keys
            }
            if self.use_imagin_robot:
                imagin_robot = torch.FloatTensor(self.imagin_robot[ind]).to(self.device)
        agent_pos = torch.FloatTensor(self.agent_pos[ind]).to(self.device)
        action = torch.FloatTensor(self.action[ind]).to(self.device)

        if not self.wo_visual:
            obs = {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
                **images,
            }
            if self.use_imagin_robot:
                obs['imagin_robot'] = imagin_robot
        else:
            obs = {
                'agent_pos': agent_pos, # T, D_pos
            }
        return {'obs':obs, 'action': action}

    def numpy_to_dict(self):
        if not self.wo_visual:
            result = {
                'point_cloud': self.point_cloud,
                'state': self.agent_pos,
                'action': self.action,
                'a_logprob': self.a_logprob,
                'reward': self.reward,
                'next_point_cloud': self.next_point_cloud,
                'next_state': self.next_agent_pos,
                'done': self.done,
                'dw': self.dw,
            }
            if self.use_legacy_image:
                result['img'] = self.image
                result['next_img'] = self.next_image
            else:
                for key in self.image_keys:
                    result[key] = self.images[key]
                    result[f'next_{key}'] = self.next_images[key]
            if self.use_imagin_robot:
                result['imagin_robot'] = self.imagin_robot
                result['next_imagin_robot'] = self.next_imagin_robot
            return result
        else:
            return {
                'state': self.agent_pos,
                'action': self.action,
                'a_logprob': self.a_logprob,
                'reward': self.reward,
                'next_state': self.next_agent_pos,
                'done': self.done,
                'dw': self.dw
            }

    
    def numpy_to_tensor(self, device=None):
        target_device = self.device if device is None else device

        def to_tensor(array):
            return torch.as_tensor(
                array, dtype=torch.float, device=target_device)

        if not self.wo_visual:
            point_cloud = to_tensor(self.point_cloud)
            images = {
                key: to_tensor(self._get_image_array(key))
                for key in self.image_keys
            }
            if self.use_imagin_robot:
                imagin_robot = to_tensor(self.imagin_robot)
        agent_pos = to_tensor(self.agent_pos)
        action = to_tensor(self.action)
        a_logprob = to_tensor(self.a_logprob)
        reward = to_tensor(self.reward)
        if not self.wo_visual:
            next_point_cloud = to_tensor(self.next_point_cloud)
            next_images = {
                key: to_tensor(self._get_image_array(key, next_obs=True))
                for key in self.image_keys
            }
            if self.use_imagin_robot:
                next_imagin_robot = to_tensor(self.next_imagin_robot)
        next_agent_pos = to_tensor(self.next_agent_pos)
        done = to_tensor(self.done)
        dw = to_tensor(self.dw)
        if not self.wo_visual:
            obs = {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
                **images,
            }
            next_obs = {
                'point_cloud': next_point_cloud, # T, 1024, 6
                'agent_pos': next_agent_pos, # T, D_pos
                **next_images,
            }
            if self.use_imagin_robot:
                obs['imagin_robot'] = imagin_robot
                next_obs['imagin_robot'] = next_imagin_robot
        else:
            obs = {
                'agent_pos': agent_pos, # T, D_pos
            }
            next_obs = {
                'agent_pos': next_agent_pos, # T, D_pos
            }

        return obs, action, a_logprob, reward, next_obs, dw, done
class IqlBuffer:
    def __init__(self, offline_data, args, shape_info,  device, wo_visual=False):
        self.use_imagin_robot = False
        for key in shape_info['obs']:
            if 'imagin_robot' in key:
                self.use_imagin_robot = True
                break
        self.wo_visual = wo_visual
        self.offline_data = offline_data
        if not wo_visual:   
            self.point_cloud = np.zeros((args.capacity, *shape_info['obs']['point_cloud']))
            self.image =  np.zeros((args.capacity, *shape_info['obs']['image']))
            if self.use_imagin_robot:
                self.imagin_robot = np.zeros((args.capacity, *shape_info['obs']['imagin_robot']))

        self.agent_pos = np.zeros((args.capacity, *shape_info['obs']['agent_pos']))
        self.action = np.zeros((args.capacity,  *shape_info['action']))
        if not self.wo_visual:
            self.next_point_cloud = np.zeros((args.capacity, *shape_info['obs']['point_cloud']))
            self.next_image = np.zeros((args.capacity, *shape_info['obs']['image']))
            if self.use_imagin_robot:
                self.next_imagin_robot = np.zeros((args.capacity, *shape_info['obs']['imagin_robot']))
        self.next_agent_pos = np.zeros((args.capacity, *shape_info['obs']['agent_pos']))
        self.reward = np.zeros((args.capacity, 1))
        self.not_done = np.zeros((args.capacity, 1))
        self.count = 0
        self.capacity = args.capacity
        self.device = device
        self.full = False
    def store(self, obs, action, reward, next_obs, done):
        if not self.wo_visual:
            self.point_cloud[self.count] = obs['point_cloud']
            self.image[self.count] = obs['image']
            if self.use_imagin_robot:
                self.imagin_robot[self.count] = obs['imagin_robot']
        self.agent_pos[self.count] = obs['agent_pos']
        self.action[self.count] = action
        self.reward[self.count] = reward
        if not self.wo_visual:
            self.next_point_cloud[self.count] = next_obs['point_cloud']
            self.next_image[self.count] = next_obs['image']
            if self.use_imagin_robot:
                self.next_imagin_robot[self.count] = next_obs['imagin_robot']
        self.next_agent_pos[self.count] = next_obs['agent_pos']
        self.not_done[self.count] = 1 - done
        self.count = (self.count + 1) % self.capacity
        self.full = self.full or self.count == 0

    def initial_with_dataset(self, dataset):
        dataset = dict_apply(dataset, lambda x: x.cpu().numpy())
        data_size = dataset['action'].shape[0]
        if not self.wo_visual:
            self.point_cloud[:data_size] = dataset['obs']['point_cloud']
            self.image[:data_size] = dataset['obs']['image']
            if self.use_imagin_robot:
                self.imagin_robot[:data_size] = dataset['obs']['imagin_robot']
        self.agent_pos[:data_size] = dataset['obs']['agent_pos']
        self.action[:data_size] = dataset['action']
        self.reward[:data_size] = dataset['reward'].squeeze(1)
        if not self.wo_visual:
            self.next_point_cloud[:data_size] = dataset['next_obs']['point_cloud']
            self.next_image[:data_size] = dataset['next_obs']['image']
            if self.use_imagin_robot:
                self.next_imagin_robot[:data_size] = dataset['next_obs']['imagin_robot']
        self.next_agent_pos[:data_size] = dataset['next_obs']['agent_pos']
        self.not_done[:data_size] = dataset['not_done'].squeeze(1)
        self.count = data_size

    def merge(self, online_batch, offline_batch):
        if offline_batch['obs']['image'].shape[-3] != 3:
            offline_batch['obs']['image'] = offline_batch['obs']['image'].permute(0, 1, 4, 3, 2)
            offline_batch['next_obs']['image'] = offline_batch['next_obs']['image'].permute(0, 1, 4, 3, 2)
        if not self.wo_visual:
            point_cloud = torch.cat([online_batch['obs']['point_cloud'], offline_batch['obs']['point_cloud']], dim=0)
            image = torch.cat([online_batch['obs']['image'], offline_batch['obs']['image']], dim=0)
            if self.use_imagin_robot:
                imagin_robot = torch.cat([online_batch['obs']['imagin_robot'], offline_batch['obs']['imagin_robot']], dim=0)
        agent_pos = torch.cat([online_batch['obs']['agent_pos'], offline_batch['obs']['agent_pos']], dim=0)
        action = torch.cat([online_batch['action'], offline_batch['action']], dim=0)
        if not self.wo_visual:
            next_point_cloud = torch.cat([online_batch['next_obs']['point_cloud'], offline_batch['next_obs']['point_cloud']], dim=0)
            next_image = torch.cat([online_batch['next_obs']['image'], offline_batch['next_obs']['image']], dim=0)
            if self.use_imagin_robot:
                next_imagin_robot = torch.cat([online_batch['next_obs']['imagin_robot'], offline_batch['next_obs']['imagin_robot']], dim=0)
        next_agent_pos = torch.cat([online_batch['next_obs']['agent_pos'], offline_batch['next_obs']['agent_pos']], dim=0)
        reward = torch.cat([online_batch['reward'], offline_batch['reward'].squeeze(1)], dim=0)
        not_done = torch.cat([online_batch['not_done'], offline_batch['not_done'].squeeze(1)], dim=0)
        if not self.wo_visual:
            if self.use_imagin_robot:
                obs = {
                    'point_cloud': point_cloud, # T, 1024, 6
                    'agent_pos': agent_pos, # T, D_pos
                    'image': image, # T, 84, 84, 3
                    'imagin_robot': imagin_robot, # T, 96, 7
                }
                next_obs = {
                    'point_cloud': next_point_cloud, # T, 1024, 6
                    'agent_pos': next_agent_pos, # T, D_pos
                    'image': next_image, # T, 84, 84, 3
                    'imagin_robot': next_imagin_robot, # T, 96, 7
                }
            else:
                obs = {
                    'point_cloud': point_cloud, # T, 1024, 6
                    'agent_pos': agent_pos, # T, D_pos
                    'image': image, # T, 84, 84, 3
                }
                next_obs = {
                    'point_cloud': next_point_cloud, # T, 1024, 6
                    'agent_pos': next_agent_pos, # T, D_pos
                    'image': next_image, # T, 84, 84, 3
                }
        else:
            obs = {
                'agent_pos': agent_pos, # T, D_pos
            }
            next_obs = {
                'agent_pos': next_agent_pos, # T, D_pos
            }
        return {'obs':obs, 'action': action, 'reward': reward, 'next_obs': next_obs, 'not_done': not_done}



    def sample(
        self, batch_size: int
    ) -> tuple:

        ind = np.random.randint(0, int(self.count), size=batch_size)
        if not self.wo_visual:
            point_cloud = torch.FloatTensor(self.point_cloud[ind]).to(self.device)
            image = torch.FloatTensor(self.image[ind]).to(self.device)
            if self.use_imagin_robot:
                imagin_robot = torch.FloatTensor(self.imagin_robot[ind]).to(self.device)
        agent_pos = torch.FloatTensor(self.agent_pos[ind]).to(self.device)
        action = torch.FloatTensor(self.action[ind]).to(self.device)
        if not self.wo_visual:

            next_point_cloud = torch.FloatTensor(self.next_point_cloud[ind]).to(self.device)
            next_image = torch.FloatTensor(self.next_image[ind]).to(self.device)
            if self.use_imagin_robot:
                next_imagin_robot = torch.FloatTensor(self.next_imagin_robot[ind]).to(self.device)
        next_agent_pos = torch.FloatTensor(self.next_agent_pos[ind]).to(self.device)
        reward = torch.FloatTensor(self.reward[ind]).to(self.device)
        not_done = torch.FloatTensor(self.not_done[ind]).to(self.device)

        if not self.wo_visual:  
            if self.use_imagin_robot:
                obs = {
                    'point_cloud': point_cloud, # T, 1024, 6
                    'agent_pos': agent_pos, # T, D_pos
                    'image': image, # T, 84, 84, 3
                    'imagin_robot': imagin_robot, # T, 96, 7
                }
                next_obs = {
                    'point_cloud': next_point_cloud, # T, 1024, 6
                    'agent_pos': next_agent_pos, # T, D_pos
                    'image': next_image, # T, 84, 84, 3
                    'imagin_robot': next_imagin_robot, # T, 96, 7
                }
            else:

                obs = {
                    'point_cloud': point_cloud, # T, 1024, 6
                    'agent_pos': agent_pos, # T, D_pos
                    'image': image, # T, 84, 84, 3
                }
                next_obs = {
                    'point_cloud': next_point_cloud, # T, 1024, 6
                    'agent_pos': next_agent_pos, # T, D_pos
                    'image': next_image, # T, 84, 84, 3
                }
        else:
            obs = {
                'agent_pos': agent_pos, # T, D_pos
            }
            next_obs = {
                'agent_pos': next_agent_pos, # T, D_pos
            }
        return {'obs':obs, 'action': action, 'reward': reward, 'next_obs': next_obs, 'not_done': not_done}
