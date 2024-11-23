import torch
import torch.nn.functional as F
from torch.optim import Adam
from agent import Agent
from model import GaussianPolicy
import numpy as np


class PPO(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)

        self.clip_epsilon = args.clip_epsilon
        self.value_coef = args.value_coef
        self.entropy_coef = args.entropy_coef
        self.num_epochs = args.num_epochs
        self.gae_lambda = args.gae_lambda
        self.action_res = args.action_res
        self.upsampled_action_res = args.action_res * args.action_res_resize

        # Policy Network (Actor)
        self.policy = GaussianPolicy(
            num_inputs,
            self.action_res,
            self.upsampled_action_res,
            args.residual,
            args.coarse2fine_bias,
        ).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=args.lr)

        # Value Network (Critic)
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, args.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_size, args.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_size, 1),
        ).to(self.device)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=args.lr)

    def select_action(self, state):
        """Select action using the policy network"""
        state = (
            (torch.FloatTensor(state) / 255.0 * 2.0 - 1.0).to(self.device).unsqueeze(0)
        )

        # Sample action from policy
        with torch.no_grad():
            action, log_prob, _, _, mask = self.policy.sample(state)
            value = self.value_net(state)

        action = action.cpu().numpy()[0]
        log_prob = log_prob.cpu().numpy()[0]
        value = value.cpu().numpy()[0]

        return action, log_prob, value

    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        lastgae = 0

        for t in reversed(range(len(rewards))):
            lastgae = (
                deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * lastgae
            )
            advantages[t] = lastgae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update_parameters(self, memory, updates):
        """Update policy and value networks using PPO"""
        # Collect and process batch data
        states, actions, rewards, next_states, dones, old_log_probs, values = (
            memory.get_batch()
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)

        # Normalize rewards
        rewards = self.reward_normalization(rewards)

        # Compute GAE and returns
        with torch.no_grad():
            next_values = self.value_net(next_states)
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        # PPO update for specified number of epochs
        for _ in range(self.num_epochs):
            # Get current policy distribution
            new_actions, new_log_probs, _, _, _ = self.policy.sample(states)

            # Compute ratio of new and old probabilities
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * advantages
            )

            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_pred = self.value_net(states)
            value_loss = F.mse_loss(value_pred, returns)

            # Entropy loss for exploration
            entropy_loss = -self.entropy_coef * new_log_probs.mean()

            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + entropy_loss

            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()

        return value_loss.item(), policy_loss.item(), entropy_loss.item()

    def save_model(self, filename):
        """Save model parameters"""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_net_state_dict": self.value_net.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            },
            filename,
        )

    def load_model(self, filename):
        """Load model parameters"""
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_net_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
