import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Upsample
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork


class DDPG(object):
    def __init__(self, num_inputs, action_space, args):
        self.args = args
        self.num_inputs = num_inputs
        self.gamma = args.gamma
        self.tau = args.tau
        self.action_res = args.action_res
        self.target_update_interval = args.target_update_interval
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.exp_upsample_list = [
            Upsample(scale_factor=i, mode="bicubic", align_corners=True)
            for i in [1, 2, 4, 8]
        ]

        # For reward normalization
        self.momentum = args.momentum
        self.mean = 0.0
        self.var = 1.0

        # Actor (Policy) Network
        self.upsampled_action_res = args.action_res * args.action_res_resize
        self.policy = GaussianPolicy(
            num_inputs,
            self.action_res,
            self.upsampled_action_res,
            args.residual,
            args.coarse2fine_bias,
        ).to(device=self.device)
        self.policy_target = GaussianPolicy(
            num_inputs,
            self.action_res,
            self.upsampled_action_res,
            args.residual,
            args.coarse2fine_bias,
        ).to(device=self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        hard_update(self.policy_target, self.policy)

        # Critic Network
        self.critic = QNetwork(
            num_inputs, self.action_res, self.upsampled_action_res, args.hidden_size
        ).to(device=self.device)
        self.critic_target = QNetwork(
            num_inputs, self.action_res, self.upsampled_action_res, args.hidden_size
        ).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, coarse_action=None, task=None):
        state = (
            (torch.FloatTensor(state) / 255.0 * 2.0 - 1.0).to(self.device).unsqueeze(0)
        )
        if coarse_action is not None:
            coarse_action = (
                torch.FloatTensor(coarse_action).to(self.device).unsqueeze(0)
            )

        # In DDPG, we directly use the mean action without sampling
        if task is None or "shapematch" not in task:
            with torch.no_grad():
                action, _, _, _, mask = self.policy.sample(state, coarse_action)
        else:
            with torch.no_grad():
                _, _, action, _, mask = self.policy.sample(state, coarse_action)
                action = torch.tanh(action)

        action = action.detach().cpu().numpy()[0]
        if coarse_action is not None:
            mask = mask.detach().cpu().numpy()[0]
        return action, mask

    def reward_normalization(self, rewards):
        batch_mean = torch.mean(rewards)
        batch_var = torch.var(rewards)
        self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
        self.var = self.momentum * self.var + (1 - self.momentum) * batch_var
        std = torch.sqrt(self.var)
        normalized_rewards = (rewards - self.mean) / (std + 1e-8)
        return normalized_rewards

    def update_parameters(self, memory, updates):
        # Sample batch from memory
        (state_batch, action_batch, reward_batch, next_state_batch, mask_batch) = (
            memory.sample(self.args.batch_size)
        )

        state_batch = (torch.FloatTensor(state_batch) / 255.0 * 2.0 - 1.0).to(
            self.device
        )
        next_state_batch = (torch.FloatTensor(next_state_batch) / 255.0 * 2.0 - 1.0).to(
            self.device
        )
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Normalize rewards
        reward_batch = self.reward_normalization(reward_batch)

        # Update Critic
        with torch.no_grad():
            if self.args.residual:
                next_original_action = self.upsample_coarse_action(next_state_batch)
                next_state_action, _, _, _, mask = self.policy_target.sample(
                    next_state_batch, next_original_action
                )
                next_state_action = mask * next_state_action + (
                    1 - mask
                ) * next_original_action.reshape(self.args.batch_size, -1)
            else:
                next_state_action, _, _, _, _ = self.policy_target.sample(
                    next_state_batch
                )

            # Get target Q value
            target_Q1, target_Q2 = self.critic_target(
                next_state_batch, next_state_action
            )
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + mask_batch * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        for params in self.critic.parameters():
            torch.nn.utils.clip_grad_norm_(params, max_norm=10)
        self.critic_optim.step()

        # Update Actor (Policy)
        if self.args.residual:
            with torch.no_grad():
                coarse_action = self.upsample_coarse_action(state_batch)
            actor_action, _, _, std, mask = self.policy.sample(
                state_batch, coarse_action
            )
            actor_action = mask * actor_action + (1 - mask) * coarse_action.reshape(
                self.args.batch_size, -1
            )
        else:
            actor_action, _, _, std, _ = self.policy.sample(state_batch)

        # Actor loss is negative of Q value
        actor_loss = -self.critic(state_batch, actor_action)[0].mean()

        if self.args.residual:
            # Add mask regularization for residual learning
            mask_regularize_loss = (
                self.args.coarse2fine_penalty
                * torch.norm(mask.reshape(mask.shape[0], -1), dim=1).mean()
                / self.args.action_res
            )
            actor_loss = actor_loss + mask_regularize_loss
        else:
            mask_regularize_loss = torch.zeros(1).to(self.device)

        # Optimize the actor
        self.policy_optim.zero_grad()
        actor_loss.backward()
        for params in self.policy.parameters():
            torch.nn.utils.clip_grad_norm_(params, max_norm=10)
        self.policy_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.policy_target, self.policy, self.tau)

        return (
            current_Q1.mean().item(),
            current_Q2.mean().item(),
            actor_loss.item(),
            0,
            0,
            torch.norm(std.reshape(std.shape[0], -1), dim=1).mean().item()
            / (self.args.action_res**2),
            mask_regularize_loss.item(),
        )

    def upsample_coarse_action(self, state_batch):
        coarse_pi, _, _, _, _ = self.coarse_policy.sample(state_batch)
        coarse_pi = coarse_pi.reshape(
            self.args.batch_size,
            2,
            self.args.coarse_action_res,
            self.args.coarse_action_res,
        )
        return self.exp_upsample_list[
            int(math.log2(self.args.action_res / self.args.coarse_action_res))
        ](coarse_pi)

    def save_model(self, filename):
        checkpoint = {
            "mean": self.mean,
            "var": self.var,
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
        }
        torch.save(checkpoint, filename + ".pth")

    def load_model(self, filename, for_train=False):
        print("Loading models from {}...".format(filename))
        checkpoint = torch.load(filename)
        mean = checkpoint.get("mean")
        var = checkpoint.get("var")
        if mean is not None:
            self.mean = mean
        if var is not None:
            self.var = var
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        if for_train:
            self.policy_optim.load_state_dict(checkpoint["policy_optim"])
            self.critic_optim.load_state_dict(checkpoint["critic_optim"])

    def load_coarse_model(self, filename, action_res):
        print("Loading coarse models from {}...".format(filename))
        self.coarse_policy = GaussianPolicy(
            self.num_inputs,
            action_res,
            self.upsampled_action_res,
            False,
            self.args.coarse2fine_bias,
        ).to(self.device)
        checkpoint = torch.load(filename)
        self.coarse_policy.load_state_dict(checkpoint["policy"])
