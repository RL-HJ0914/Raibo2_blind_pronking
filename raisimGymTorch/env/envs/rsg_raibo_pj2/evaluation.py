import time

import numpy as np
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.rsg_raibo_pj2 import RaisimGymRaiboPJ2
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import torch
import argparse
import collections
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
eval_mode = True
cfg['environment']['render'] = True
if eval_mode:
    cfg['environment']['num_envs'] = 600
    cfg['environment']['num_threads'] = 32
    device = 'cuda'
else:
    cfg['environment']['num_envs'] = 1
    cfg['environment']['num_threads'] = 1
    device = 'cpu'

env = VecEnv(RaisimGymRaiboPJ2(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
# env.seed(0)
# env.set_command(0)  # ensures that the initial command is zero

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

weight_path = args.weight

command = np.zeros(3, dtype=np.float32)

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'
    print("Loaded weight from {}\n".format(weight_path))
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.GRU_MLP_Actor(ob_dim - 3,
                                            cfg['architecture']['hidden_dim'],
                                            cfg['architecture']['mlp_shape'],
                                            act_dim,
                                            env.num_envs,
                                            device)
    loaded_graph.load_state_dict(torch.load(weight_path, map_location=device)['actor_architecture_state_dict'])
    loaded_graph.init_hidden()

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()
    for i in range(int(iteration_number)):
        env.curriculum_callback()

    while True:
        data_tags = env.get_step_data_tag()
        data_size = 0
        data_mean = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
        data_square_sum = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
        data_min = np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)
        data_max = -np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)
        env.reset()
        for step in range(n_steps):
            obs = env.observe(False)
            with torch.no_grad():
                action_ll = loaded_graph.forward(torch.from_numpy(np.expand_dims(obs, axis=0)).to(device)).squeeze(dim=0)
                # _, dones = env.step_visualize(action_ll.cpu().detach().numpy())
                _, dones = env.step(action_ll.cpu().detach().numpy())
                # plt.pause(cfg['environment']['control_dt'])
                data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)
                if np.sum(dones) > 0:
                    arg_dones = np.argwhere(dones).flatten()
                    loaded_graph.init_by_done(arg_dones)
        if eval_mode:
            for data_id in range(len(data_tags)):
                print(data_tags[data_id] + ": " + str(data_mean[data_id]))
