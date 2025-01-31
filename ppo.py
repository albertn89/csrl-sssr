import os
import math
import torch
from torch.optim import Adam
from torch.nn import Upsample
import torch.nn.functional as F
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork
from agent import Agent

class PPO(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)
        
        self.policy = GaussianPolicy(
            num_inputs,
            self.action_res,
            self.upsampled_action_res,
            args.residual,
            args.coarse2fine_bias).to(device=self.device)
        
        
        self.optimizer = Adam(self.policy.parameters(), lr=args.lr)
        self.value_loss_fn = torch.nn.MSELoss()
        self.gamma = args.gamma
        self.tau = args.tau
        self.clip_epsilon = args.clip_epsilon
        self.value_coef = args.value_coef
        self.entropy_coef = args.entropy_coef

    
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
        states, actions, rewards, next_states, log_probs, dones = memory.sample()
        
        # Compute advantages and returns
        values = self.policy.evaluate(states)
        next_values = self.policy.evaluate(next_states)
        advantages = rewards + (1 - dones) * self.gamma * next_values - values.detach()
        
        returns = advantages + values  # Target for value function
        
        for _ in range(updates):
            new_log_probs, entropy = self.policy.evaluate_actions(states, actions)
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - log_probs.detach())
            
            # Surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = self.value_loss_fn(values, returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
