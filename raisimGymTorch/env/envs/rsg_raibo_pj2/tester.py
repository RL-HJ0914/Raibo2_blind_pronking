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
import pygame
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
cfg['environment']['num_envs'] = 1
cfg['environment']['num_threads'] = 1
cfg['environment']['render'] = True
cfg['environment']['randomization']['test_mode'] = False
test_mode = cfg['environment']['randomization']['test_mode']
cfg['environment']['randomization']['terrain_randomization'] = not(test_mode)
cfg['environment']['randomization']['observation_randomization'] = not(test_mode)
cfg['environment']['randomization']['joint_friction_randomization'] = not(test_mode)
cfg['environment']['randomization']['gain_randomization'] = not(test_mode)
cfg['environment']['curriculum']['initial_factor'] = 0.2

env = VecEnv(RaisimGymRaiboPJ2(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
# env.seed(0)
# env.set_command(0)  # ensures that the initial command is zero

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

weight_path = "/home/hyungjun/raisim_workspace/Raibo2_blind_bounding/raisimGymTorch/data/raibo2_blind_bounding/2025-01-16-02-28-07/full_10000.pt"
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

command = np.zeros(3, dtype=np.float32)
gc = np.zeros(19, dtype=np.float32)
gv = np.zeros(18, dtype=np.float32)
info = np.zeros(62, dtype=np.float32)

# plotting
infoBag = []
gcBag = []
gvBag = []


if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))

    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    for joystick in joysticks:
        print(joystick.get_name())

    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.GRU_MLP_Actor(ob_dim - 3,
                                            cfg['architecture']['hidden_dim'],
                                            cfg['architecture']['mlp_shape'],
                                            act_dim,
                                            env.num_envs,
                                            'cpu')
    loaded_graph.load_state_dict(torch.load(weight_path, map_location='cpu')['actor_architecture_state_dict'])
    loaded_graph.init_hidden()

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()
    for i in range(int(iteration_number)):
        env.curriculum_callback()
    env.reset()
    env.set_command(np.zeros(3, dtype=np.float32))

    for step in range(total_steps*2000):
        for event in pygame.event.get():  # User did something.
            if event.type == pygame.JOYBUTTONDOWN:  # If user clicked close.
                if event.button == 0:
                    env.terrain_change()
                    env.reset()
                    print("change env")
                elif event.button == 1:
                    env.reset()
                    print("env reset")
        if len(joysticks) > 0:
            command[0] = -4 * joysticks[0].get_axis(1)
            command[1] = - joysticks[0].get_axis(0)
            command[2] = - joysticks[0].get_axis(3)

        if test_mode:
            if step < 1000:
                command[0] = 1.0
            elif step < 2000:
                command[0] = 2.0
            elif step < 3000:
                command[0] = 3.0
            elif step < 4000:
                command[0] = 4.0
            elif step < 5000:
                command[0] = 2.0
            elif step < 5200:
                command[0] = 0.0
            elif step < 6200:
                command[1] = 1.0
            elif step < 7200:
                command[1] = 0.0
                command[2] = 1.0
            elif step < 7400:
                command[1] = 0.0
                command[2] = 0.0
            else:
                a = 1
        env.set_command(command)

        obs = env.observe(False)
        with torch.no_grad():
            action_ll = loaded_graph.forward(torch.from_numpy(np.expand_dims(obs, axis=0)).cpu()).squeeze(dim=0)
            if test_mode:
                _, dones = env.step(action_ll.cpu().detach().numpy())
                plt.pause(cfg['environment']['control_dt'])
            else:
                _, dones = env.step_visualize(action_ll.cpu().detach().numpy())
            if np.sum(dones) > 0:
                arg_dones = np.argwhere(dones).flatten()
                loaded_graph.init_by_done(arg_dones)

        # plotting
        env.get_state(gc, gv)
        env.set_command(command)
        env.get_logging_info(info)
        infoBag.append(info.copy())
        gcBag.append(gc.copy())
        gvBag.append(gv.copy())

    env.turn_off_visualization()

    import mplcursors
    infoBag_np = np.array(infoBag)
    gcBag_np = np.array(gcBag)
    gvBag_np = np.array(gvBag)

    plot_start_idx = 0

    # body z
    fig = plt.figure(0)
    plt.plot(gcBag_np[plot_start_idx:, 2])

    # torque
    fig0 = plt.figure(1)
    ax4 = fig0.add_subplot(141)
    ax5 = fig0.add_subplot(142)
    ax6 = fig0.add_subplot(143)
    ax7 = fig0.add_subplot(144)
    ax4.plot(infoBag_np[plot_start_idx:, 12])
    ax4.plot(infoBag_np[plot_start_idx:, 13])
    ax4.plot(infoBag_np[plot_start_idx:, 14])
    ax5.plot(infoBag_np[plot_start_idx:, 15])
    ax5.plot(infoBag_np[plot_start_idx:, 16])
    ax5.plot(infoBag_np[plot_start_idx:, 17])
    ax6.plot(infoBag_np[plot_start_idx:, 18])
    ax6.plot(infoBag_np[plot_start_idx:, 19])
    ax6.plot(infoBag_np[plot_start_idx:, 20])
    ax7.plot(infoBag_np[plot_start_idx:, 21])
    ax7.plot(infoBag_np[plot_start_idx:, 22])
    ax7.plot(infoBag_np[plot_start_idx:, 23])

    # swing leg trajectories
    import mplcursors
    infoBag_np = np.array(infoBag)
    gcBag_np = np.array(gcBag)
    gvBag_np = np.array(gvBag)
    plot_start_idx = 0
    fig0 = plt.figure(1)
    ax0 = fig0.add_subplot(411)
    ax1 = fig0.add_subplot(412)
    ax2 = fig0.add_subplot(413)
    ax3 = fig0.add_subplot(414)
    ax0.plot(infoBag_np[plot_start_idx:, 0])
    ax0.plot(infoBag_np[plot_start_idx:, 4])
    ax0.plot(infoBag_np[plot_start_idx:, 8])
    ax0.plot(infoBag_np[plot_start_idx:, 36])
    ax0.plot(infoBag_np[plot_start_idx:, 32])
    ax1.plot(infoBag_np[plot_start_idx:, 1])
    ax1.plot(infoBag_np[plot_start_idx:, 5])
    ax1.plot(infoBag_np[plot_start_idx:, 9])
    ax1.plot(infoBag_np[plot_start_idx:, 37])
    ax1.plot(infoBag_np[plot_start_idx:, 33])
    ax2.plot(infoBag_np[plot_start_idx:, 2])
    ax2.plot(infoBag_np[plot_start_idx:, 6])
    ax2.plot(infoBag_np[plot_start_idx:, 10])
    ax2.plot(infoBag_np[plot_start_idx:, 38])
    ax2.plot(infoBag_np[plot_start_idx:, 34])
    ax3.plot(infoBag_np[plot_start_idx:, 3])
    ax3.plot(infoBag_np[plot_start_idx:, 7])
    ax3.plot(infoBag_np[plot_start_idx:, 11])
    ax3.plot(infoBag_np[plot_start_idx:, 39])
    ax3.plot(infoBag_np[plot_start_idx:, 35])
    mplcursors.cursor()

    plt.plot(infoBag_np[plot_start_idx:, 36:40])
    plt.plot(infoBag_np[plot_start_idx:, 8:12])
    infoBag_np[plot_start_idx:, 8:12].min()

    # joint_vel - torque trajectories
    fig1 = plt.figure(2)
    ax4 = fig1.add_subplot(141)
    ax5 = fig1.add_subplot(142)
    ax6 = fig1.add_subplot(143)
    ax7 = fig1.add_subplot(144)
    ax4.plot(gvBag_np[plot_start_idx:, 6], infoBag_np[plot_start_idx:, 12])
    ax4.plot(gvBag_np[plot_start_idx:, 7], infoBag_np[plot_start_idx:, 13])
    ax4.plot(gvBag_np[plot_start_idx:, 8], infoBag_np[plot_start_idx:, 14])
    ax5.plot(gvBag_np[plot_start_idx:, 9], infoBag_np[plot_start_idx:, 15])
    ax5.plot(gvBag_np[plot_start_idx:, 10], infoBag_np[plot_start_idx:, 16])
    ax5.plot(gvBag_np[plot_start_idx:, 11], infoBag_np[plot_start_idx:, 17])
    ax6.plot(gvBag_np[plot_start_idx:, 12], infoBag_np[plot_start_idx:, 18])
    ax6.plot(gvBag_np[plot_start_idx:, 13], infoBag_np[plot_start_idx:, 19])
    ax6.plot(gvBag_np[plot_start_idx:, 14], infoBag_np[plot_start_idx:, 20])
    ax7.plot(gvBag_np[plot_start_idx:, 15], infoBag_np[plot_start_idx:, 21])
    ax7.plot(gvBag_np[plot_start_idx:, 16], infoBag_np[plot_start_idx:, 22])
    ax7.plot(gvBag_np[plot_start_idx:, 17], infoBag_np[plot_start_idx:, 23])

    # joint_vel
    fig1 = plt.figure(3)
    ax4 = fig1.add_subplot(141)
    ax5 = fig1.add_subplot(142)
    ax6 = fig1.add_subplot(143)
    ax7 = fig1.add_subplot(144)
    ax4.plot(gvBag_np[plot_start_idx:, 6])
    ax4.plot(gvBag_np[plot_start_idx:, 7])
    ax4.plot(gvBag_np[plot_start_idx:, 8])
    ax5.plot(gvBag_np[plot_start_idx:, 9])
    ax5.plot(gvBag_np[plot_start_idx:, 10])
    ax5.plot(gvBag_np[plot_start_idx:, 11])
    ax6.plot(gvBag_np[plot_start_idx:, 12])
    ax6.plot(gvBag_np[plot_start_idx:, 13])
    ax6.plot(gvBag_np[plot_start_idx:, 14])
    ax7.plot(gvBag_np[plot_start_idx:, 15])
    ax7.plot(gvBag_np[plot_start_idx:, 16])
    ax7.plot(gvBag_np[plot_start_idx:, 17])

    # torque
    fig1 = plt.figure(4)
    ax4 = fig1.add_subplot(141)
    ax5 = fig1.add_subplot(142)
    ax6 = fig1.add_subplot(143)
    ax7 = fig1.add_subplot(144)
    ax4.plot(infoBag_np[plot_start_idx:, 12])
    ax4.plot(infoBag_np[plot_start_idx:, 13])
    ax4.plot(infoBag_np[plot_start_idx:, 14])
    ax5.plot(infoBag_np[plot_start_idx:, 15])
    ax5.plot(infoBag_np[plot_start_idx:, 16])
    ax5.plot(infoBag_np[plot_start_idx:, 17])
    ax6.plot(infoBag_np[plot_start_idx:, 18])
    ax6.plot(infoBag_np[plot_start_idx:, 19])
    ax6.plot(infoBag_np[plot_start_idx:, 20])
    ax7.plot(infoBag_np[plot_start_idx:, 21])
    ax7.plot(infoBag_np[plot_start_idx:, 22])
    ax7.plot(infoBag_np[plot_start_idx:, 23])

    # air time
    fig2 = plt.figure(5)
    ax8 = fig2.add_subplot(411)
    ax9 = fig2.add_subplot(412)
    ax10 = fig2.add_subplot(413)
    ax11 = fig2.add_subplot(414)
    # ax12 = fig2.add_subplot(515)
    ax8.plot(infoBag_np[plot_start_idx:, 24])
    # ax8.plot(infoBag_np[plot_start_idx:, 28])
    ax9.plot(infoBag_np[plot_start_idx:, 25])
    # ax9.plot(infoBag_np[plot_start_idx:, 29])
    ax10.plot(infoBag_np[plot_start_idx:, 26])
    # ax10.plot(infoBag_np[plot_start_idx:, 30])
    ax11.plot(infoBag_np[plot_start_idx:, 27])
    # ax11.plot(infoBag_np[plot_start_idx:, 31])
    # ax12.plot(infoBag_np[plot_start_idx:, 32:36])

    # stance time
    fig2 = plt.figure(6)
    ax8 = fig2.add_subplot(411)
    ax9 = fig2.add_subplot(412)
    ax10 = fig2.add_subplot(413)
    ax11 = fig2.add_subplot(414)
    # ax12 = fig2.add_subplot(515)
    # ax8.plot(infoBag_np[plot_start_idx:, 24])
    ax8.plot(infoBag_np[plot_start_idx:, 28])
    # ax9.plot(infoBag_np[plot_start_idx:, 25])
    ax9.plot(infoBag_np[plot_start_idx:, 29])
    # ax10.plot(infoBag_np[plot_start_idx:, 26])
    ax10.plot(infoBag_np[plot_start_idx:, 30])
    # ax11.plot(infoBag_np[plot_start_idx:, 27])
    ax11.plot(infoBag_np[plot_start_idx:, 31])

    # command tracking
    fig3 = plt.figure(6)
    ax13 = fig3.add_subplot(311)
    ax14 = fig3.add_subplot(312)
    ax15 = fig3.add_subplot(313)
    t = np.arange(0., infoBag_np.shape[0] * 0.01, 0.01)
    ax13.plot(t, infoBag_np[plot_start_idx:, 40])
    ax13.plot(t, infoBag_np[plot_start_idx:, 43])
    ax14.plot(t, infoBag_np[plot_start_idx:, 41])
    ax14.plot(t, infoBag_np[plot_start_idx:, 44])
    ax15.plot(t, infoBag_np[plot_start_idx:, 42])
    ax15.plot(t, infoBag_np[plot_start_idx:, 45])

    # GRF
    fig0 = plt.figure(6)
    ax0 = fig0.add_subplot(411)
    ax1 = fig0.add_subplot(412)
    ax2 = fig0.add_subplot(413)
    ax3 = fig0.add_subplot(414)
    ax0.plot(infoBag_np[plot_start_idx:, 32])
    ax0.plot(infoBag_np[plot_start_idx:, 46])
    ax1.plot(infoBag_np[plot_start_idx:, 33])
    ax1.plot(infoBag_np[plot_start_idx:, 47])
    ax2.plot(infoBag_np[plot_start_idx:, 34])
    ax2.plot(infoBag_np[plot_start_idx:, 48])
    ax3.plot(infoBag_np[plot_start_idx:, 35])
    ax3.plot(infoBag_np[plot_start_idx:, 49])

    # undesiredGRF
    fig0 = plt.figure(7)
    ax0 = fig0.add_subplot(411)
    ax1 = fig0.add_subplot(412)
    ax2 = fig0.add_subplot(413)
    ax3 = fig0.add_subplot(414)
    ax0.plot(infoBag_np[plot_start_idx:, 50])
    ax0.plot(infoBag_np[plot_start_idx:, 54])
    ax1.plot(infoBag_np[plot_start_idx:, 51])
    ax1.plot(infoBag_np[plot_start_idx:, 55])
    ax2.plot(infoBag_np[plot_start_idx:, 52])
    ax2.plot(infoBag_np[plot_start_idx:, 56])
    ax3.plot(infoBag_np[plot_start_idx:, 53])
    ax3.plot(infoBag_np[plot_start_idx:, 57])

    # stride
    fig0 = plt.figure(8)
    ax0 = fig0.add_subplot(411)
    ax1 = fig0.add_subplot(412)
    ax2 = fig0.add_subplot(413)
    ax3 = fig0.add_subplot(414)
    ax0.plot(infoBag_np[plot_start_idx:, 58])
    ax1.plot(infoBag_np[plot_start_idx:, 59])
    ax2.plot(infoBag_np[plot_start_idx:, 60])
    ax3.plot(infoBag_np[plot_start_idx:, 61])
    mplcursors.cursor()