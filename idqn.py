import collections
import random
import sys
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ma_gym.wrappers import Monitor
import argparse
from utils.file_utils import *
from utils.learning_utils import *
from envs import make_one_env, make_vec_envs, make_vec_envs_2
from wrappers import RecordEpisodeStatistics, SquashDones, GlobalizeReward, FlattenObservation
from collections import deque

USE_WANDB = True  # if enabled, logs data on wandb server
NUM_PARALLEL = 8

def _squash_info(info, eval = False):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")

    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        new_info[key] = mean
    return new_info

class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append((np.ones(done.shape) - done).tolist())

        return torch.tensor(s_lst, dtype=torch.float), \
               torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), \
               torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(128, 64),
                                                                    nn.ReLU(),
                                                                    nn.Linear(64, action_space[agent_i].n)))

    def forward(self, obs):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        for agent_i in range(self.num_agents):
            q_values[agent_i] = getattr(self, 'agent_{}'.format(agent_i))(obs[:, agent_i, :]).unsqueeze(1)

        return torch.cat(q_values, dim=1)

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1], out.shape[2],))
        action[mask] = torch.randint(0, out.shape[3], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=3).float()
        action = action.swapaxes(1, 2)
        return action


def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10):
    for _ in range(update_iter):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        # print(s.size())
        # if(s.dim() == 4):
        #     s = s.swapaxes(1, 2)
        #     s_prime = s_prime.swapaxes(1, 2)
        q_out = q(s).swapaxes(1, 2)
        q_a = q_out.gather(3, a.unsqueeze(-1).long()).squeeze(-1)
        max_q_prime = q_target(s_prime).swapaxes(1, 2).max(dim=3)[0]
        # NOTE: I am repeating the done flag for all agents, this might be problematic
        done_mask = done_mask.unsqueeze(2).repeat(1, 1, r.size()[2])
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(env, num_episodes, q, env_name):
    all_infos = []
    state = env.reset()
    while len(all_infos) < num_episodes:
        action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon=0)[0].data.cpu().numpy().tolist()
        next_state, reward, done, infos = env.step(action)
        state = next_state       
        for info in infos:
                if("predator_prey" in env_name or "PredatorPrey" in env_name or "TrafficJunction" in env_name or "MarlGrid" in env_name):
                    if('episode_reward' in info.keys()):
                        all_infos.append(info)
                else:
                    if info:
                        all_infos.append(info)

    squashed = _squash_info(all_infos, eval = True)
    test_score = squashed['episode_reward'].sum()
    return test_score


def main(algo, seed, env_configs, env_name, lr, gamma, batch_size, buffer_limit, log_interval, max_updates,
         max_epsilon, min_epsilon, test_episodes, warm_up_steps, update_iter):

    """
    Unused code to use vectorized environment
    """
    # num_processes  = 12 
    # wrappers = tuple([RecordEpisodeStatistics, SquashDones, FlattenObservation])
    # use_dummy_venv = False


    # envs = make_vec_envs(
    #     env_configs['env_name'],
    #     env_configs,
    #     seed,
    #     dummy_vecenv,
    #     num_processes,
    #     env_configs["time_limit"] if "time_limit" in env_configs.keys() else time_limit,
    #     wrappers,
    #     algorithm["device"],
    #     env_properties= algorithm['env_properties']
    # )


    # Setting seed
    set_seed(seed)

    # Init env 
    envs = make_vec_envs_2(env_name, env_configs, seed, NUM_PARALLEL)
    test_envs = make_vec_envs_2(env_name, env_configs, seed, NUM_PARALLEL)
    memory = ReplayBuffer(buffer_limit)

    q = QNet(envs.observation_space, envs.action_space)
    q_target = QNet(envs.observation_space, envs.action_space)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)
    num_updates = 0 
    # (num agent, num_parallel, num_feature)
    state = envs.reset()
    all_infos = deque(maxlen = NUM_PARALLEL)
    while(num_updates != max_updates):
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (num_updates / (0.4 * max_updates)))
        dones = [False for _ in range(NUM_PARALLEL)]
        action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon)[0].data.cpu().numpy().tolist()
        next_state, reward, dones, infos = envs.step(action)
        memory.put((state, action, (np.array(reward)), next_state, np.array(dones, dtype=int)))

        # print("dones: {}".format(dones))
        # print("{}, {}, {}".format(len(all_infos), infos[0].keys(), reward))
        # print("stepping")
        sys.stdout.flush()

        for info in infos:
            if("predator_prey" in env_name or "PredatorPrey" in env_name or "TrafficJunction" in env_name or "MarlGrid" in env_name):
                if('episode_reward' in info.keys()):
                    all_infos.append(info)
            else:
                if info:
                    all_infos.append(info)

        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, gamma, batch_size, update_iter)
            num_updates += 1

        if num_updates % log_interval == 0 and len(all_infos) > 1:
            squashed = _squash_info(all_infos)
            train_episode_reward = squashed['episode_reward'].sum()
            q_target.load_state_dict(q.state_dict())
            test_episode_reward = test(test_envs, test_episodes, q, env_name)
            print("#{:<10}/{} updates , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, eps : {:.1f}"
                  .format(num_updates, max_updates, train_episode_reward, test_episode_reward, memory.size(), epsilon))
            if USE_WANDB:
                wandb.log({'update': num_updates, 'test-score': test_episode_reward,
                           'buffer-size': memory.size(), 'epsilon': epsilon, 'train-score':train_episode_reward})
            all_infos.clear()

    env.close()
    test_env.close()

    # score = np.zeros(env.n_agents)
    # for episode_i in range(max_episodes):
    #     epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.4 * max_episodes)))
    #     state = env.reset()
    #     done = [False for _ in range(env.n_agents)]
    #     # print("step")
    #     sys.stdout.flush()
    #     while not all(done):
    #         action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon)[0].data.cpu().numpy().tolist()
    #         next_state, reward, done, info = env.step(action)
    #         memory.put((state, action, (np.array(reward)).tolist(), next_state, np.array(done, dtype=int).tolist()))
    #         score += np.array(reward)
    #         state = next_state

    #     if memory.size() > warm_up_steps:
    #         train(q, q_target, memory, optimizer, gamma, batch_size, update_iter)

    #     if episode_i % log_interval == 0 and episode_i != 0:
    #         q_target.load_state_dict(q.state_dict())
    #         test_score = test(test_env, test_episodes, q)
    #         print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, eps : {:.1f}"
    #               .format(episode_i, max_episodes, sum(score / log_interval), test_score, memory.size(), epsilon))
    #         if USE_WANDB:
    #             wandb.log({'episode': episode_i, 'test-score': test_score,
    #                        'buffer-size': memory.size(), 'epsilon': epsilon, 'train-score': sum(score / log_interval)})
    #         score = np.zeros(env.n_agents)

    # env.close()
    # test_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IDQN')
    parser.add_argument('--config_path')
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    kwargs = parse_yaml(args.config_path)
    kwargs['seed'] = args.seed

    print(kwargs)

    if USE_WANDB:
        import wandb

        wandb.init(project='marl_reset', config={'algo': 'idqn', **kwargs}, monitor_gym=False)

    main(**kwargs)
