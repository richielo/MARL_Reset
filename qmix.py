import sys
import collections
import gym
import numpy as np
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

    def sample_chunk(self, batch_size, chunk_size):
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                s, a, r, s_prime, done = self.buffer[chunk_step]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                done_lst.append(done)

        n_agents, obs_size = len(s_lst[0]), len(s_lst[0][0][0])
        return torch.tensor(s_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, NUM_PARALLEL, obs_size), \
               torch.tensor(a_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, NUM_PARALLEL), \
               torch.tensor(r_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, NUM_PARALLEL), \
               torch.tensor(s_prime_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, NUM_PARALLEL, obs_size), \
               torch.tensor(done_lst, dtype=torch.float).view(batch_size, chunk_size, NUM_PARALLEL, 1)

    def size(self):
        return len(self.buffer)


class MixNet(nn.Module):
    def __init__(self, observation_space, hidden_dim=32, hx_size=64, recurrent=False):
        super(MixNet, self).__init__()
        state_size = sum([_.shape[0] for _ in observation_space])
        self.hidden_dim = hidden_dim
        self.hx_size = hx_size
        self.n_agents = len(observation_space)
        self.recurrent = recurrent

        hyper_net_input_size = state_size
        if self.recurrent:
            self.gru = nn.GRUCell(state_size, self.hx_size)
            hyper_net_input_size = self.hx_size
        self.hyper_net_weight_1 = nn.Linear(hyper_net_input_size, self.n_agents * hidden_dim)
        self.hyper_net_weight_2 = nn.Linear(hyper_net_input_size, hidden_dim)

        self.hyper_net_bias_1 = nn.Linear(hyper_net_input_size, hidden_dim)
        self.hyper_net_bias_2 = nn.Sequential(nn.Linear(hyper_net_input_size, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, 1))

    def forward(self, q_values, observations, hidden):
        batch_size, n_agents, n_para, obs_size = observations.shape
        state = observations.swapaxes(1, 2).contiguous().view(batch_size, n_para, n_agents * obs_size)
        x = state
        if self.recurrent:
            x_s = []
            h_s = []
            for para_idx in range(NUM_PARALLEL):
                h = self.gru(x[:, para_idx, :], hidden[:, para_idx, :])
                x_s.append(h.clone())
                h_s.append(h.clone())
            hidden = torch.stack(h_s)
            x = torch.stack(x_s)
            # hidden = self.gru(x, hidden)
            # x = hidden

        weight_1 = torch.abs(self.hyper_net_weight_1(x))
        """
        Original: (batch, num_agents)
        W1: (batch, hidden, num_agents)
        """
        weight_1 = weight_1.view(batch_size, n_para, self.hidden_dim, n_agents)
        if self.recurrent:
            bias_1 = self.hyper_net_bias_1(x).unsqueeze(-1).swapaxes(0, 1)
            weight_2 = torch.abs(self.hyper_net_weight_2(x)).swapaxes(0, 1)
            bias_2 = self.hyper_net_bias_2(x).swapaxes(0, 1)
        else:
            bias_1 = self.hyper_net_bias_1(x).unsqueeze(-1)
            weight_2 = torch.abs(self.hyper_net_weight_2(x))
            bias_2 = self.hyper_net_bias_2(x)
        # NOTE: need to flatten and reflatten to do bmm
        flattned_w_1 = weight_1.view(batch_size * n_para,  self.hidden_dim, n_agents)
        flattned_q_vals = q_values.view(batch_size * n_para, q_values.size(2)).unsqueeze(-1)
        flat_bmm = torch.bmm(flattned_w_1, flattned_q_vals)
        # print("{}, {}, {}".format(flat_bmm.size(), flattned_w_1.size(), flattned_q_vals.size()))
        unflattened_bmm = flat_bmm.view(batch_size, n_para, flat_bmm.size(1), flat_bmm.size(2))
        # print("{}, {}".format(unflattened_bmm.size(), bias_1.size()))
        # exit()
        x = unflattened_bmm + bias_1
        x = torch.relu(x)
        x = (weight_2.unsqueeze(-1) * x).sum(dim=2) + bias_2
        return x, hidden.swapaxes(0, 1) if self.recurrent else hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, NUM_PARALLEL, self.hx_size))
    
    def init_hidden_non_para(self, batch_size = 1):
        return torch.zeros((batch_size, self.hx_size))


class QNet(nn.Module):
    def __init__(self, observation_space, action_space, recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.recurrent = recurrent
        self.hx_size = 32
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            setattr(self, 'agent_feature_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                            nn.ReLU(),
                                                                            nn.Linear(128, self.hx_size),
                                                                            nn.ReLU()))
            if recurrent:
                setattr(self, 'agent_gru_{}'.format(agent_i), nn.GRUCell(self.hx_size, self.hx_size))
            setattr(self, 'agent_q_{}'.format(agent_i), nn.Linear(self.hx_size, action_space[agent_i].n))

    def forward(self, obs, hidden):
        q_values = [torch.empty(obs.shape[0], NUM_PARALLEL, )] * self.num_agents
        next_hidden = [torch.empty(obs.shape[0], NUM_PARALLEL, 1, self.hx_size, )] * self.num_agents
        for agent_i in range(self.num_agents):
            x = getattr(self, 'agent_feature_{}'.format(agent_i))(obs[:, agent_i, :, :])
            x_output = torch.empty(x.size())
            if self.recurrent:
                for para_idx in range(NUM_PARALLEL):
                    x_output[:, para_idx, :] = getattr(self, 'agent_gru_{}'.format(agent_i))(x[:, para_idx, :], hidden[:, agent_i, para_idx, :])
                    next_hidden[agent_i][:, para_idx, :, :] = x_output[:, para_idx, :] .unsqueeze(1)
                # x = getattr(self, 'agent_gru_{}'.format(agent_i))(x, hidden[:, agent_i, :, :])
                # next_hidden[agent_i] = x.unsqueeze(1)
            q_values[agent_i] = getattr(self, 'agent_q_{}'.format(agent_i))(x_output).unsqueeze(1)
     
        cat_q_values = torch.cat(q_values, dim=1).swapaxes(1, 2)
        # print(torch.cat(next_hidden, dim=2).size())
        # print("here 2")
        return cat_q_values, torch.cat(next_hidden, dim=2).swapaxes(1, 2)

    def sample_action(self, obs, hidden, epsilon):
        out, hidden = self.forward(obs, hidden)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1], out.shape[2],))
        action[mask] = torch.randint(0, out.shape[3], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=3).float()
        action = action.swapaxes(1, 2)
        return action, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, NUM_PARALLEL, self.hx_size))
    
    def init_hidden_non_para(self, batch_size = 1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))

def train(q, q_target, mix_net, mix_net_target, memory, optimizer, gamma, batch_size, update_iter=10, chunk_size=10,
          grad_clip_norm=5):
    _chunk_size = chunk_size if q.recurrent else 1
    for _ in range(update_iter):
        s, a, r, s_prime, done = memory.sample_chunk(batch_size, _chunk_size)
        r = r.swapaxes(2, 3)
        hidden = q.init_hidden(batch_size)
        target_hidden = q_target.init_hidden(batch_size)
        mix_net_target_hidden = mix_net_target.init_hidden(batch_size)
        mix_net_hidden = [torch.empty_like(mix_net_target_hidden) for _ in range(_chunk_size + 1)]
        mix_net_hidden[0] = mix_net_target.init_hidden(batch_size)
        loss = 0
        for step_i in range(_chunk_size):
            q_out, hidden = q(s[:, step_i, :, :, :], hidden.clone())
            q_a = q_out.gather(3, a[:, step_i, :, :].unsqueeze(-1).long().swapaxes(1, 2)).squeeze(-1)
            pred_q, next_mix_net_hidden = mix_net(q_a, s[:, step_i, :, :, :], mix_net_hidden[step_i].clone())
            # next_mix_net_hidden = next_mix_net_hidden.swapaxes(0, 1)

            max_q_prime, target_hidden = q_target(s_prime[:, step_i, :, :, :], target_hidden.detach())
            max_q_prime = max_q_prime.max(dim=3)[0].squeeze(-1)
            q_prime_total, mix_net_target_hidden = mix_net_target(max_q_prime, s_prime[:, step_i, :, :, :],
                                                                  mix_net_target_hidden.detach())
            # mix_net_target_hidden = mix_net_target_hidden.swapaxes(0, 1)
            target_q = r[:, step_i, :].sum(dim=2, keepdims=True) + (gamma * q_prime_total * (1 - done[:, step_i]))
            # print(target_q.size())
            # print(r.size())
            # print(done.size())
            # print(q_prime_total.size())
            loss += F.smooth_l1_loss(pred_q, target_q.detach())

 
            done_mask = done[:, step_i].squeeze(-1).bool()
            hidden = hidden.swapaxes(1, 2)
            hidden[done_mask] = q.init_hidden_non_para(len(hidden[done_mask]))
            hidden = hidden.swapaxes(1, 2)

            target_hidden = target_hidden.swapaxes(1, 2)
            target_hidden[done_mask] = q_target.init_hidden_non_para(len(target_hidden[done_mask]))
            target_hidden = target_hidden.swapaxes(1, 2)
            # print("{}, {}, {} ,{}".format(hidden.size(), target_hidden.size(), mix_net_hidden[step_i + 1].size(), mix_net_target_hidden.size()))

            mix_net_hidden[step_i + 1][~done_mask] = next_mix_net_hidden[~done_mask]
            mix_net_hidden[step_i + 1][done_mask] = mix_net.init_hidden_non_para(len(mix_net_hidden[step_i][done_mask]))
            mix_net_target_hidden[done_mask] = mix_net_target.init_hidden_non_para(len(mix_net_target_hidden[done_mask]))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), grad_clip_norm, norm_type=2)
        torch.nn.utils.clip_grad_norm_(mix_net.parameters(), grad_clip_norm, norm_type=2)
        optimizer.step()

def test(env, num_episodes, q, env_name):
    all_infos = []
    state = env.reset()
    while len(all_infos) < num_episodes:
        action, hidden = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon=0)
        next_state, reward, done, info = env.step(action[0].data.cpu().numpy().tolist())
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
         max_epsilon, min_epsilon, test_episodes, warm_up_steps, update_iter, chunk_size,
         update_target_interval, recurrent):
    # Setting seed
    set_seed(seed)

    # Init env 
    envs = make_vec_envs_2(env_name, env_configs, seed, NUM_PARALLEL)
    test_envs = make_vec_envs_2(env_name, env_configs, seed, NUM_PARALLEL)
    memory = ReplayBuffer(buffer_limit)


    # Init env 
    # env = make_one_env(env_name, env_configs, seed)
    # test_env = make_one_env(env_name, env_configs, seed)
    # memory = ReplayBuffer(buffer_limit)

    # create networks
    q = QNet(envs.observation_space, envs.action_space, recurrent)
    q_target = QNet(envs.observation_space, envs.action_space, recurrent)
    q_target.load_state_dict(q.state_dict())

    mix_net = MixNet(envs.observation_space, recurrent=recurrent)
    mix_net_target = MixNet(envs.observation_space, recurrent=recurrent)
    mix_net_target.load_state_dict(mix_net.state_dict())

    optimizer = optim.Adam([{'params': q.parameters()}, {'params': mix_net.parameters()}], lr=lr)

    num_updates = 0 
    # (num agent, num_parallel, num_feature)
    state = envs.reset()
    with torch.no_grad():
        hidden = q.init_hidden()
    all_infos = deque(maxlen = NUM_PARALLEL)
    while(num_updates != max_updates):
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (num_updates / (0.4 * max_updates)))
        dones = [False for _ in range(NUM_PARALLEL)]
        action, hidden = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon)
        action = action[0].data.cpu().numpy().tolist()
        sys.stdout.flush()
        next_state, reward, dones, infos = envs.step(action)
        memory.put((state, action, (np.array(reward)).tolist(), next_state, dones))
        # print("stepping")
        sys.stdout.flush()

        # Reset hidden if an env instance is done
        for d_idx in range(dones.shape[0]):
            if(dones[d_idx]):
                hidden[:, d_idx] = q.init_hidden()[:, d_idx]

        for info in infos:
            if("predator_prey" in env_name or "PredatorPrey" in env_name or "TrafficJunction" in env_name or "MarlGrid" in env_name):
                if('episode_reward' in info.keys()):
                    all_infos.append(info)
            else:
                if info:
                    all_infos.append(info)

        if memory.size() > warm_up_steps:
            train(q, q_target, mix_net, mix_net_target, memory, optimizer, gamma, batch_size, update_iter, chunk_size)
            num_updates += 1

        if num_updates % update_target_interval:
            q_target.load_state_dict(q.state_dict())
            mix_net_target.load_state_dict(mix_net.state_dict())

        if num_updates % log_interval == 0 and num_updates != 0:
            squashed = _squash_info(all_infos)
            train_episode_reward = squashed['episode_reward'].sum()
            test_episode_reward = test(test_envs, test_episodes, q), env_name
            print("#{:<10}/{} updates , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, eps : {:.1f}"
                  .format(num_updates, max_updates, train_episode_reward, test_episode_reward, memory.size(), epsilon))
            if USE_WANDB:
                    wandb.log({'update': num_updates, 'test-score': test_episode_reward,
                            'buffer-size': memory.size(), 'epsilon': epsilon, 'train-score':train_episode_reward})
            all_infos.clear()

    envs.close()
    test_envs.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QMIX')
    parser.add_argument('--config_path')
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    # parser = argparse.ArgumentParser(description='Qmix')
    # parser.add_argument('--env-name', required=False, default='ma_gym:Checkers-v0')
    # parser.add_argument('--seed', type=int, default=1, required=False)
    # parser.add_argument('--no-recurrent', action='store_true')
    # parser.add_argument('--max-episodes', type=int, default=10000, required=False)

    # # Process arguments
    # args = parser.parse_args()

    # kwargs = {'env_name': args.env_name,
    #           'lr': 0.001,
    #           'batch_size': 32,
    #           'gamma': 0.99,
    #           'buffer_limit': 50000,
    #           'update_target_interval': 20,
    #           'log_interval': 100,
    #           'max_episodes': args.max_episodes,
    #           'max_epsilon': 0.9,
    #           'min_epsilon': 0.1,
    #           'test_episodes': 5,
    #           'warm_up_steps': 2000,
    #           'update_iter': 10,
    #           'chunk_size': 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
    #           'recurrent': not args.no_recurrent}

    kwargs = parse_yaml(args.config_path)
    kwargs['seed'] = args.seed

    print(kwargs)

    if USE_WANDB:
        import wandb

        wandb.init(project='marl_reset', config={'algo': 'qmix', **kwargs})

    main(**kwargs)
