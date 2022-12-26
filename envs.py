import os

import gym
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from gym.spaces.box import Box
from gym.wrappers import Monitor
import easydict

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from wrappers import TimeLimit, Monitor, FlattenObservation, RecordEpisodeStatistics, SquashDones, GlobalizeReward

from mod_envs.pp_wrapper import *
from mod_envs.tj_wrapper import *
# from mod_envs.overcook_wrapper import OverCookedEnv
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
import lbforaging

# from vmas import make_env as vmas_make_env

WRAPPERS = [
    RecordEpisodeStatistics,
    SquashDones,
    FlattenObservation
]

def make_one_env(env_name, env_configs, seed, rank = 0):
    def _thunk():
        if('FullCoopPredatorPreyWrapper' in env_name):
            env = FullCoopPredatorPreyWrapper(centralized = False, grid_shape = (env_configs['grid_size'], env_configs['grid_size']),
                    n_agents = env_configs['num_agents'],
                    n_preys = env_configs['num_preys'],
                    step_cost = env_configs['step_cost'],
                    max_steps = env_configs['time_limit'],
                    prey_capture_reward = env_configs['prey_capture_reward'],
                    penalty = env_configs['penalty'],
                    other_agent_visible= env_configs['other_agent_visible'],
                    prey_move_probs = tuple(env_configs['prey_move_probs'])
            )
            env.seed(seed + rank)
            for a_space in env.action_space:
                a_space.seed(seed + rank)
        elif('TrafficJunction' in env_name):
            # Not using curriculum learning, so rate min equal to rate max
            env = TrafficJunctionWrapper(
                centralized = False,
                dim = env_configs['dim'],
                vision = env_configs['vision'],
                add_rate_min = env_configs['add_rate_max'],
                add_rate_max = env_configs['add_rate_max'],
                curr_start = 0,
                curr_end = 0,
                difficulty = env_configs['difficulty'],
                n_agents = env_configs['num_agents'],
                max_steps = env_configs['time_limit']
            )
            env.seed(seed + rank)
        elif('Overcook' in env_name):
            # env = OverCookedEnv(scenario = env_configs['env'], episode_length = env_configs['time_limit'])
            # base_mdp = OvercookedGridworld.from_layout_name(env_configs['env'])
            # env = OvercookedEnv.from_mdp(base_mdp, horizon = env_configs['time_limit'])

            mdp = OvercookedGridworld.from_layout_name(env_configs['env'])
            base_env = OvercookedEnv.from_mdp(mdp, horizon = env_configs['time_limit'])
            config_dict = {'base_env' : base_env, 'featurize_fn' : base_env.featurize_state_mdp}
            env = gym.make('Overcooked-v0', **config_dict)
            # env = env.custom_init(base_env, base_env.featurize_state_mdp)
            # print(env.observation_space.shape)
            print("initialized overcook")
        else:
            env = gym.make(env_name)
            env.seed(seed + rank)
        env = TimeLimit(env, max_episode_steps = env_configs['time_limit'])
        for wrapper in WRAPPERS:
            env = wrapper(env)
        return env
    return _thunk


class MADummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        agents = len(self.observation_space)
        # change this because we want >1 reward
        self.buf_rews = np.zeros((self.num_envs, agents), dtype=np.float32)

def make_env(env_id, env_configs, seed, rank, time_limit, wrappers, monitor_dir):
    def _thunk():
        if('FullCoopPredatorPreyWrapper' in env_id):
            env = FullCoopPredatorPreyWrapper(centralized = False, grid_shape = (env_configs['grid_size'], env_configs['grid_size']),
                  n_agents = env_configs['num_agents'],
                  n_preys = env_configs['num_preys'],
                  step_cost = env_configs['step_cost'],
                  max_steps = env_configs['time_limit'],
                  prey_capture_reward = env_configs['prey_capture_reward'],
                  penalty = env_configs['penalty'],
                  other_agent_visible= env_configs['other_agent_visible'],
                  prey_move_probs = tuple(env_configs['prey_move_probs'])
            )
            env.seed(seed + rank)
            for a_space in env.action_space:
                a_space.seed(seed + rank)
            # env.action_space.seed(seed + rank)
        elif('TrafficJunction' in env_id):
            # Not using curriculum learning, so rate min equal to rate max
            env = TrafficJunctionWrapper(
                centralized = False,
                dim = env_configs['dim'],
                vision = env_configs['vision'],
                add_rate_min = env_configs['add_rate_max'],
                add_rate_max = env_configs['add_rate_max'],
                curr_start = 0,
                curr_end = 0,
                difficulty = env_configs['difficulty'],
                n_agents = env_configs['num_agents'],
                max_steps = env_configs['time_limit']
            )
            env.seed(seed + rank)
            # env.action_space.seed(seed + rank)
        else:
            env = gym.make(env_id)
            env.seed(seed + rank)
            env.action_space.seed(seed + rank)


        if time_limit:
            env = TimeLimit(env, time_limit)

        # Remove the flatten observation wrapper
        for wrapper in wrappers:
            env = wrapper(env)

        if monitor_dir:
            env = Monitor(env, monitor_dir, lambda ep: int(ep==0), force=True, uid=str(rank))

        return env

    return _thunk


def make_vec_envs(
    env_name, env_configs, seed, dummy_vecenv, parallel, time_limit, wrappers, device, monitor_dir = None
):
    envs = [
        make_env(env_name, env_configs, seed, i, time_limit, wrappers, monitor_dir) for i in range(parallel)
    ]

    if dummy_vecenv or len(envs) == 1 or monitor_dir:
        envs = MADummyVecEnv(envs)
    else:
        try:
            envs = SubprocVecEnv(envs, start_method="fork")
        except:
            envs = SubprocVecEnv(envs, start_method="spawn")

    envs = VecPyTorch(envs, device, env_name)
    return envs


def make_vec_envs_2(
    env_name, env_configs, seed, num_parallel):
    envs = [
        make_one_env(env_name, env_configs, seed, i) for i in range(num_parallel)
    ]

    try:
        envs = SubprocVecEnv(envs, start_method="fork")
    except:
        envs = SubprocVecEnv(envs, start_method="spawn")

    return envs

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device, env_name):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        self.env_name = env_name
        # print(env_name)
        # TODO: Fix data types


    def reset(self):
        # num_agents x num_processes x obs_size
        obs = self.venv.reset()

        return [torch.from_numpy(o).to(self.device) for o in obs]

    def step_async(self, actions):
        actions = [a.squeeze().cpu().numpy() for a in actions]
        actions = list(zip(*actions))
        return self.venv.step_async(actions)

    def step_wait(self):

        # We need (num_agent, num_process, num_features), original pp environment gives (num_processes, num_agent, num_features)
        obs, rew, done, info = self.venv.step_wait()

        return (
            [torch.from_numpy(o).float().to(self.device) for o in obs],
            torch.from_numpy(rew).float().to(self.device),
            torch.from_numpy(done).float().to(self.device),
            info,
        )