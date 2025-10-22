from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .storage import RolloutStorage


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 smoothness_loss_coef=0.5,
                 smoothness_loss_end=10000,
                 policy_learning_rate=5e-4,
                 value_learning_rate=1e-3,
                 lr_scheduler_rate=0.99999,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 shuffle_batch=True):

        # PPO components
        self.actor = actor
        self.critic = critic
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape, actor.action_shape, device)

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.policy_optimizer = optim.Adam([*self.actor.parameters()], lr=policy_learning_rate)
        self.value_optimizer = optim.Adam([*self.critic.parameters()], lr=value_learning_rate)
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, 1, lr_scheduler_rate)
        self.value_scheduler = optim.lr_scheduler.StepLR(self.value_optimizer, 1, lr_scheduler_rate)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.smoothness_loss_coef = smoothness_loss_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.smoothness_loss_end = smoothness_loss_end

        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None

    def act(self, actor_obs):
        self.actor_obs = actor_obs
        with torch.no_grad():
            self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device))
        return self.actions

    def step(self, value_obs, rews, dones):
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, self.actor.action_mean, self.actor.distribution.std_np, rews, dones,
                                     self.actions_log_prob)

    def update(self, value_obs, log_this_iteration, update):
        last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))

        # Learning step
        self.storage.compute_returns(last_values.to(self.device), self.critic, self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_smoothness_loss, infos = self._train_step(log_this_iteration, update)
        self.storage.clear()

        if log_this_iteration:
            self.log({**locals(), 'it': update})

    def log(self, variables):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        mean_std = self.actor.distribution.std.mean()

        self.writer.add_scalar('PPO/value_function', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar('PPO/surrogate', variables['mean_surrogate_loss'], variables['it'])
        self.writer.add_scalar('PPO/entropy', variables['mean_entropy_loss'], variables['it'])
        self.writer.add_scalar('PPO/smoothness', variables['mean_smoothness_loss'], variables['it'])
        self.writer.add_scalar('PPO/mean_noise_std', mean_std.item(), variables['it'])
        self.writer.add_scalar('PPO/learning_rate', self.policy_scheduler.get_last_lr()[0], variables['it'])

    def _train_step(self, log_this_iteration, update):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_smoothness_loss = 0
        for epoch in range(self.num_learning_epochs):
            self.actor.architecture.init_hidden()

            actor_obs_batch = self.storage.actor_obs_tc
            critic_obs_batch = self.storage.critic_obs_tc
            actions_batch = self.storage.actions_tc.view(-1, self.storage.actions_tc.size(-1))
            old_sigma_batch = self.storage.sigma_tc.view(-1 ,self.storage.sigma_tc.size(-1))
            old_mu_batch = self.storage.mu_tc.view(-1, self.storage.mu_tc.size(-1))
            current_values_batch = self.storage.values_tc.view(-1 ,1)
            advantages_batch = self.storage.advantages_tc.view(-1, 1)
            returns_batch = self.storage.returns_tc.view(-1, 1)
            old_actions_log_prob_batch = self.storage.actions_log_prob_tc.view(-1, 1)

            (actions_log_prob_batch, entropy_batch), mean_actions_batch = self.actor.evaluate(actor_obs_batch, actions_batch)
            value_batch = self.critic.evaluate(critic_obs_batch)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = current_values_batch + (value_batch - current_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # smoothness loss
            smoothness_loss = 0.5 * (mean_actions_batch[1:] - mean_actions_batch[:-1]).pow(2).mean() + \
                              (mean_actions_batch[2:] - 2 * mean_actions_batch[1:-1] + mean_actions_batch[:-2]).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.smoothness_loss_coef * smoothness_loss * min(update, self.smoothness_loss_end) / self.smoothness_loss_end

            # Gradient step
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
            self.policy_optimizer.step()
            self.value_optimizer.step()

            if log_this_iteration:
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy_loss += entropy_batch.mean().item()
                mean_smoothness_loss += smoothness_loss.item()

        self.policy_scheduler.step()
        self.value_scheduler.step()

        if log_this_iteration:
            num_updates = self.num_learning_epochs
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            mean_entropy_loss /= num_updates
            mean_smoothness_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_smoothness_loss, locals()