# task specification
task_name = "raibo2_blind_bounding"

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.rsg_raibo_pj2 import RaisimGymRaiboPJ2
from raisimGymTorch.env.bin.rsg_raibo_pj2 import NormalSampler
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import argparse
import datetime
import wandb
import faulthandler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
wandb_log=cfg['wandb_log']
ip= cfg["environment"]["ip"]
reward_list = cfg['environment']['reward']
reward_coeff = []
for key, value in reward_list.items():
    reward_coeff.append(value)

cfg['environment']['curriculum']['target_smoothness_end'] = int(cfg['environment']['curriculum']['target_learning_episode_end'] / 50000 * np.log(3.25 / 0.05) * 10000)
std_schedule_rate = 0.0001 * 50000 / cfg['environment']['curriculum']['target_learning_episode_end']
lr_schedule_rate = np.power(10, -2 / cfg['environment']['curriculum']['target_learning_episode_end'])

# create environment from the configuration file
env = VecEnv(RaisimGymRaiboPJ2(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
             cfg['environment'])

# shortcuts
ob_dim = env.num_obs
value_ob_dim = env.num_value_obs
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

actor = ppo_module.Actor(ppo_module.GRU_MLP_Actor(ob_dim - 3,
                                                  cfg['architecture']['hidden_dim'],
                                                  cfg['architecture']['mlp_shape'],
                                                  act_dim,
                                                  env.num_envs),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           3.5,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)

critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['mlp3_shape'], nn.LeakyReLU, value_ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/" + task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp",
                                       task_path + "/runner.py", task_path + "/RaiboController.hpp",
                                       home_path + "/raisimGymTorch/algo/ppo/module.py",
                                       home_path + "/raisimGymTorch/algo/ppo/ppo.py"])

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=32,
              gamma=0.99,
              lam=0.95,
              num_mini_batches=1,
              policy_learning_rate=5e-4,
              value_learning_rate=5e-4,
              lr_scheduler_rate=lr_schedule_rate,
              max_grad_norm=0.5,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=True,
              entropy_coef=0.005,
              smoothness_loss_coef=0.025,
              value_loss_coef=0.5,
              smoothness_loss_end=cfg['environment']['curriculum']['target_smoothness_end']
              )

iteration_number = 0
time_str=datetime.datetime.now().strftime("%m-%d-%H")

if (wandb_log):
    wandb.init(project="Raibo2_blind_bounding", name=time_str+"at"+str(ip),
               config={
                   "num_envs ": cfg["environment"]["num_envs"],
                   "max time ": cfg["environment"]["max_time"]
               }
               )
    wandb.config.update(wandb.config)

def reset():
    env.reset()
    ppo.actor.architecture.init_hidden()

def by_terminate(dones):
    if np.sum(dones) > 0:
        arg_dones = np.argwhere(dones).flatten()
        ppo.actor.architecture.init_by_done(arg_dones)


if mode == 'retrain':
    iteration_number = load_param(weight_path, env, actor, critic, ppo.policy_optimizer, ppo.value_optimizer, ppo.policy_scheduler, ppo.value_scheduler, saver.data_dir)
    # load observation scaling from files of pre-trained model
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'
    env.full_load_scaling(weight_dir, iteration_number, env.num_envs * iteration_number * n_steps)
    for curriculum_update in range(int(iteration_number)):
        env.curriculum_callback()

for update in range(iteration_number, cfg['environment']['curriculum']['target_learning_episode_end'] + 1):
    start = time.time()
    reset()
    reward_ll_sum = 0
    done_sum = 0
    env.turn_off_visualization()
    if update % cfg['environment']['iteration_per_save'] == 0:
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'policy_optimizer_state_dict': ppo.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': ppo.value_optimizer.state_dict(),
            'policy_scheduler_state_dict': ppo.policy_scheduler.state_dict(),
            'value_scheduler_state_dict': ppo.value_scheduler.state_dict(),
        }, saver.data_dir + "/full_" + str(update) + '.pt')
        env.save_scaling(saver.data_dir, str(update))

    if update % cfg['environment']['eval_every_n'] == 0:
        if update % cfg['environment']['iteration_per_record'] == 0:
            print("Visualizing and evaluating the current policy")
            env.turn_on_visualization()
            env.start_video_recording(
                datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "policy_" + str(update) + '.mp4')
        data_tags = env.get_step_data_tag()
        data_size = 0
        data_mean = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
        data_square_sum = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
        data_min = np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)
        data_max = -np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)

        for step in range(n_steps):
            with torch.no_grad():
                obs = env.observe(False)
                actions = actor.forward(torch.from_numpy(np.expand_dims(obs, axis=0)).to(device))

                if update != 0 and update % cfg['environment']['iteration_per_record'] == 0:
                    reward, dones = env.step_visualize(actions)
                else:
                    reward, dones = env.step(actions)

                done_sum = done_sum + np.sum(dones)
                reward_ll_sum = reward_ll_sum + np.sum(reward)
                data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)
                by_terminate(dones)

        average_ll_performance = reward_ll_sum / total_steps
        average_dones = done_sum / total_steps

        if update % cfg['environment']['iteration_per_record'] == 0:
            env.stop_video_recording()
            env.turn_off_visualization()
        reset()
        reward_ll_sum = 0
        done_sum = 0
        data_std = np.sqrt((data_square_sum - data_size * data_mean * data_mean) / (data_size - 1 + 1e-16))
        if(wandb_log):
            for data_id in range(len(data_tags)):
                data_log=dict()
                data_log[data_tags[data_id]+'/mean'] = data_mean[data_id]
                data_log[data_tags[data_id]+'/std'] = data_std[data_id]
                data_log[data_tags[data_id]+'/min'] = data_min[data_id]
                data_log[data_tags[data_id]+'/max'] = data_max[data_id]
                wandb.log(data_log, step=update)

    # actual training
    for step in range(n_steps):
        with torch.no_grad():
            obs = env.observe(update < 10000)
            value_obs = env.get_value_obs(update < 10000)
            action = ppo.act(np.expand_dims(obs, axis=0))
            reward, dones = env.step(action)
            ppo.step(value_obs=value_obs, rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_ll_sum = reward_ll_sum + np.sum(reward)
            by_terminate(dones)

    # take st step to get value obs
    value_obs = env.get_value_obs(update < 10000)
    ppo.update(value_obs=np.expand_dims(value_obs, axis=0),
               log_this_iteration=update % cfg['environment']['iteration_per_log'] == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    actor.update()

    env.curriculum_callback()

    if update % cfg['environment']['iteration_per_log'] == 0 and wandb_log:
        Training_log = dict()
        Training_log['Training/average_reward'] = average_ll_performance
        Training_log['Training/dones'] = average_dones
        Training_log['Training/actor_std'] = actor.distribution.std.mean()
        wandb.log(Training_log, step=update)

    end = time.time()

    print('----------------------------------------------------')
    print('at ip ',ip)
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('----------------------------------------------------\n')
