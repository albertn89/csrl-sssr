import os
import math
import torch
from torch.optim import Adam
from torch.nn import Upsample
import torch.nn.functional as F
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork
from agent import Agent


class DDPG(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)

    def select_action(self, state,  task=None):
        state = (torch.FloatTensor(state) / 255.0 * 2.0 - 1.0).to(self.device).unsqueeze(0)

        if task is None or "shapematch" not in task:
            action, _, _, _, mask = self.policy.sample(state)
        else:
            _, _, action, _, mask = self.policy.sample(state)
            action = torch.tanh(action)
        action = action.detach().cpu().numpy()[0]

        return action, None

    def update_parameters(self, memory, updates):
        return super().update_parameters(memory, updates)