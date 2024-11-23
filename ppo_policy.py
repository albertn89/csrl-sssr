import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from model import GaussianPolicy, weights_init_


class PPOPolicy(GaussianPolicy):
    """Extended GaussianPolicy with PPO-specific functionality"""

    def __init__(self, num_inputs, action_res, max_action_res, residual=False, bias=0):
        super().__init__(num_inputs, action_res, max_action_res, residual, bias)

        # Add value function network
        self.value = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

        # Initialize the value network weights
        self.value.apply(weights_init_)

    def get_value(self, state):
        """Compute the value function estimate for a given state."""
        x = self.obs(state)
        x = x.reshape(x.shape[0], -1)  # Flatten the features
        value = self.value(x)
        return value

    def evaluate_actions(self, state, action):
        """Evaluate actions for PPO update.
        Returns action log probs, entropy, and values."""
        if self.residual:
            mean, std, mask = self.forward(state, coarse_action)
            action = mask * action + (1 - mask) * coarse_action.reshape(
                action.shape[0], -1
            )
        else:
            mean, std, _ = self.forward(state)

        normal = Normal(mean, std)
        log_probs = normal.log_prob(action).sum(1, keepdim=True)
        entropy = normal.entropy().mean()

        return action, log_probs, entropy
