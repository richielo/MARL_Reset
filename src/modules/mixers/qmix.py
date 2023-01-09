import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

    def reset(self, mode):
        # NOTE: Becasue of how qmix layers are structured
        # last - lasy layers of each branch (i.e., w_1 and w_final) with biases
        # full - everything  
        if(mode == 'last'):
            if getattr(args, "hypernet_layers", 1) == 1:
                # With only one layer, this actually resets everything except for self.V which only its last layer is reset
                torch.nn.init.xavier_uniform_(list(self.V.children())[-1].weight)
                torch.nn.init.xavier_uniform_(self.hyper_b_1.weight)
                torch.nn.init.xavier_uniform_(self.hyper_w_1.weight)
                torch.nn.init.xavier_uniform_(self.hyper_w_final.weight)
            elif getattr(args, "hypernet_layers", 1) == 2:
                torch.nn.init.xavier_uniform_(list(self.V.children())[-1].weight)
                torch.nn.init.xavier_uniform_(self.hyper_b_1.weight)
                torch.nn.init.xavier_uniform_(list(self.hyper_w_1.children())[-1].weight)
                torch.nn.init.xavier_uniform_(list(self.hyper_w_final.children())[-1].weight)
        elif(mode == 'full'):
            if getattr(args, "hypernet_layers", 1) == 1:
                for layer in list(self.V.children()):
                     torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.xavier_uniform_(self.hyper_b_1.weight)
                torch.nn.init.xavier_uniform_(self.hyper_w_1.weight)
                torch.nn.init.xavier_uniform_(self.hyper_w_final.weight)
            elif getattr(args, "hypernet_layers", 1) == 2:
                for layer in list(self.V.children()):
                     torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.xavier_uniform_(self.hyper_b_1.weight)
                for layer in list(self.hyper_w_1.children()):
                     torch.nn.init.xavier_uniform_(layer.weight)
                for layer in list(self.hyper_w_final.children()):
                     torch.nn.init.xavier_uniform_(layer.weight)
        else:
            raise NotImplementedError() 
