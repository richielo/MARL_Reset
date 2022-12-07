from ma_gym.envs.predator_prey import PredatorPrey
import numpy as np
from gym import spaces
import gym

class MultiAgentObservationSpace(list):
    def __init__(self, agents_observation_space):
        for x in agents_observation_space:
            assert isinstance(x, gym.spaces.space.Space)

        super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space

    def sample(self):
        """ samples observations for each agent from uniform distribution"""
        return [agent_observation_space.sample() for agent_observation_space in self._agents_observation_space]

    def contains(self, obs):
        """ contains observation """
        for space, ob in zip(self._agents_observation_space, obs):
            if not space.contains(ob):
                return False
        else:
            return True

class FullCoopPredatorPrey(PredatorPrey):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(5, 5), n_agents=2, n_preys=1, prey_move_probs=(0.175, 0.175, 0.175, 0.175, 0.3),
                 full_observable=False, penalty=-0.5, step_cost=-0.01, prey_capture_reward=5, max_steps=100):
        super().__init__(grid_shape,n_agents,  n_preys, prey_move_probs, full_observable, penalty, step_cost, prey_capture_reward, max_steps)

    #     mask_size = np.prod(self._agent_view_mask)
    #     self._agent_view_mask = (3, 3)
    #     self._obs_high = np.array([1., 1.] + [1.] * mask_size + [1.0], dtype=np.float32)
    #     self._obs_low = np.array([0., 0.] + [0.] * mask_size + [0.0], dtype=np.float32)
    #     if self.full_observable:
    #         self._obs_high = np.tile(self._obs_high, self.n_agents)
    #         self._obs_low = np.tile(self._obs_low, self.n_agents)
    #     self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

    # def get_agent_obs(self):
    #     _obs = []
    #     for agent_i in range(self.n_agents):
    #         pos = self.agent_pos[agent_i]
    #         _agent_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)] # coordinates

    #         # check if prey is in the view area
    #         _prey_pos = np.zeros(self._agent_view_mask)  # prey location in neighbour
    #         for row in range(max(0, pos[0] - 1), min(pos[0] + 1 + 1, self._grid_shape[0])):
    #             for col in range(max(0, pos[1] - 1), min(pos[1] + 1 + 1, self._grid_shape[1])):
    #                 if PRE_IDS['prey'] in self._full_obs[row][col]:
    #                     _prey_pos[row - (pos[0] - 1), col - (pos[1] - 1)] = 1  # get relative position for the prey loc.

    #         _agent_i_obs += _prey_pos.flatten().tolist()  # adding prey pos in observable area
    #         _agent_i_obs += [self._step_count / self._max_steps]  # adding time
    #         _obs.append(_agent_i_obs)

    #     if self.full_observable:
    #         _obs = np.array(_obs).flatten().tolist()
    #         _obs = [_obs for _ in range(self.n_agents)]
    #     return _obs

    def step(self, agents_action):
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self._PredatorPrey__update_agent_pos(agent_i, action)
        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                predator_neighbour_count, n_i = self._neighbour_agents(self.prey_pos[prey_i])

                if predator_neighbour_count >= 1:
                    num_possible_neighbors = len(self._PredatorPrey__get_neighbour_coordinates(self.prey_pos[prey_i]))
                    _reward = self._penalty if predator_neighbour_count < num_possible_neighbors else self._prey_capture_reward
                    self._prey_alive[prey_i] = (predator_neighbour_count < num_possible_neighbors)

                    # for agent_i in range(self.n_agents):
                    #     rewards[agent_i] += _reward
                    for agent_i in n_i:
                        rewards[agent_i] += _reward

                prey_move = None
                if self._prey_alive[prey_i]:
                    # 5 trails : we sample next move and check if prey (smart) doesn't go in neighbourhood of predator
                    for _ in range(5):
                        _move = self.np_random.choice(len(self._prey_move_probs), 1, p=self._prey_move_probs)[0]
                        if self._neighbour_agents(self._PredatorPrey__next_pos(self.prey_pos[prey_i], _move))[0] == 0:
                            prey_move = _move
                            break
                    prey_move = 4 if prey_move is None else prey_move  # default is no-op(4)

                self._PredatorPrey__update_prey_pos(prey_i, prey_move)

        if (self._step_count >= self._max_steps) or (True not in self._prey_alive):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        return self.get_agent_obs(), rewards, self._agent_dones, {'prey_alive': self._prey_alive}

PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
    'wall': 'W',
    'empty': '0'
}