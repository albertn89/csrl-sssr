import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from agent import Agent
from ppo_policy import PPOPolicy  # Import the new PPOPolicy class


class PPO(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)

        self.action_res = args.action_res
        self.upsampled_action_res = args.action_res * args.action_res_resize

        # Use PPOPolicy instead of GaussianPolicy
        self.policy = PPOPolicy(
            num_inputs,
            self.action_res,
            self.upsampled_action_res,
            args.residual,
            args.coarse2fine_bias,
        ).to(device=self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=args.lr)

        # Training parameters
        self.clip_epsilon = args.clip_epsilon
        self.epochs = args.ppo_epochs
        self.value_loss_coef = args.value_coef
        self.entropy_coef = args.entropy_coef

    def select_action(self, state, task=None):
        state = (
            (torch.FloatTensor(state) / 255.0 * 2.0 - 1.0).to(self.device).unsqueeze(0)
        )

        with torch.no_grad():
            if task is None or "shapematch" not in task:
                action, _, _, _, mask = self.policy.sample(state)
            else:
                _, _, action, _, mask = self.policy.sample(state)
                action = torch.tanh(action)

        action = action.detach().cpu().numpy()[0]
        return action, None

    def update_parameters(self, memory, updates):
        # Get batch of experiences
        states, actions, rewards, next_states, dones = memory.sample(
            self.args.batch_size
        )

        # Convert to tensors and normalize states
        states = (torch.FloatTensor(states) / 255.0 * 2.0 - 1.0).to(self.device)
        next_states = (torch.FloatTensor(next_states) / 255.0 * 2.0 - 1.0).to(
            self.device
        )
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # Normalize rewards using running statistics
        rewards = self.reward_normalization(rewards)

        # Compute advantages and returns
        with torch.no_grad():
            next_values = self.policy.get_value(next_states)
            current_values = self.policy.get_value(states)

            # GAE (Generalized Advantage Estimation)
            advantages = torch.zeros_like(rewards).to(self.device)
            gae = 0
            for t in reversed(range(rewards.shape[0])):
                if t == rewards.shape[0] - 1:
                    next_value = next_values[t]
                else:
                    next_value = current_values[t + 1]

                delta = (
                    rewards[t]
                    + self.gamma * next_value * (1 - dones[t])
                    - current_values[t]
                )
                gae = delta + self.gamma * self.tau * (1 - dones[t]) * gae
                advantages[t] = gae

            returns = advantages + current_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_value_loss = 0
        total_policy_loss = 0
        total_entropy = 0

        for _ in range(self.epochs):
            # Get current policy outputs
            _, log_probs, entropy = self.policy.evaluate_actions(states, actions)
            values = self.policy.get_value(states)

            # Compute ratio of new and old probabilities
            ratio = torch.exp(log_probs)

            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * advantages
            )

            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Total loss
            loss = (
                policy_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy
            )

            # Update policy
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()

            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item()

        # Average losses over epochs
        avg_value_loss = total_value_loss / self.epochs
        avg_policy_loss = total_policy_loss / self.epochs
        avg_entropy = total_entropy / self.epochs

        return avg_value_loss, 0, avg_policy_loss, avg_entropy, self.alpha, 0, 0

    def save_model(self, filename):
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "mean": self.mean,
                "var": self.var,
            },
            filename + ".pth",
        )

    def load_model(self, filename, for_train=False):
        print("Loading models from {}...".format(filename))
        checkpoint = torch.load(filename + ".pth")
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if for_train:
            self.policy_optimizer.load_state_dict(
                checkpoint["policy_optimizer_state_dict"]
            )
            self.mean = checkpoint["mean"]
            self.var = checkpoint["var"]
