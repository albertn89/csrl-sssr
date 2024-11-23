import torch
import torch.nn.functional as F
from torch.optim import Adam
from agent import Agent
from replay_memory import ReplayMemory
from model import GaussianPolicy


class PPO(Agent):
    def __init__(self, num_inputs, action_space, args):
        super(PPO, self).__init__(num_inputs, action_space, args)

        self.policy = GaussianPolicy(
            num_inputs,
            args.action_res,
            args.action_res_resize,
            args.residual,
            args.coarse2fine_bias,
        ).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=args.lr)

        self.value_function = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, args.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_size, 1),
        ).to(self.device)
        self.value_optimizer = Adam(self.value_function.parameters(), lr=args.lr)

        self.memory = ReplayMemory(args.replay_size, args.seed, args.batch_size)
        self.clip_epsilon = args.clip_epsilon
        self.value_coef = args.value_coef
        self.entropy_coef = args.entropy_coef

    def collect_trajectories(self, env, num_steps):
        state = env.reset()
        for _ in range(num_steps):
            action, log_prob = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            mask = 1 - int(done)
            self.memory.push(state, action, reward, next_state, mask, log_prob)
            state = next_state
            if done:
                state = env.reset()

    def compute_gae(self, rewards, masks, values, next_values, gamma, tau):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step] + gamma * next_values[step] * masks[step] - values[step]
            )
            gae = delta + gamma * tau * masks[step] * gae
            advantages.insert(0, gae)
        return advantages

    def update_parameters(self, num_updates):
        for _ in range(num_updates):
            states, actions, rewards, next_states, masks, old_log_probs = (
                self.memory.sample()
            )

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
            masks = torch.FloatTensor(masks).to(self.device).unsqueeze(1)
            old_log_probs = (
                torch.FloatTensor(old_log_probs).to(self.device).unsqueeze(1)
            )

            values = self.value_function(states)
            next_values = self.value_function(next_states)
            advantages = self.compute_gae(
                rewards, masks, values, next_values, self.gamma, self.tau
            )
            advantages = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)

            returns = advantages + values

            # Update policy
            new_log_probs, entropy = self.policy.evaluate_actions(states, actions)
            ratios = (new_log_probs - old_log_probs).exp()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * advantages
            )
            policy_loss = (
                -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
            )

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Update value function
            value_loss = F.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob = self.policy.sample(state)
        return action.cpu().numpy()[0], log_prob.cpu().item()
