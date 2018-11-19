from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
from environment import create_env
from utils import setup_logger
from model import A3C_CONV, A3C_MLP
from player_util import Agent
from torch.autograd import Variable
import time
import logging
import gym
from torcs_wrapper import TorcsWrapper
import matplotlib.pyplot as plt

def test(args, shared_model, num_train):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env),
                 r'{0}{1}_log'.format(args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)

    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, None, args, None)
    player.gpu_id = gpu_id
    if args.model == 'MLP':
        player.model = A3C_MLP(25, 1, 1)
            # player.env.observation_space.shape[0], player.env.action_space, args.stack_frames)
    # if args.model == 'CONV':
    #     player.model = A3C_CONV(args.stack_frames, player.env.action_space)

    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
    player.model.eval()

    max_score = 0
    max_mean_score = 0

    reward_sum_list = []
    mean_reward_list = []
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())

        reward_sum = 0
        env = TorcsWrapper(port=4000, noisy=True, throttle=0.20, control_dim=1, k=2.0)
        player.env = env
        state = env.reset()
        player.state = torch.from_numpy(state).float()
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.state = player.state.cuda()

        while True:
            player.action_test()
            reward_sum += player.reward
            if player.done:
                break

        if player.done:
            player.eps_len = 0
            player.clear_actions()

            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))

            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    state_to_save = player.model.state_dict()
            else:
                state_to_save = player.model.state_dict()

            if reward_sum >= max_score:
                max_score = reward_sum
                torch.save(state_to_save, '{0}{1}-max.dat'.format(args.save_model_dir, args.env))

            if reward_mean >= max_mean_score:
                max_mean_score = reward_mean
                torch.save(state_to_save, '{0}{1}-mean.dat'.format(args.save_model_dir, args.env))

            torch.save(state_to_save, '{0}{1}-last.dat'.format(args.save_model_dir, args.env))

            reward_sum_list.append(reward_sum)
            mean_reward_list.append(reward_mean)

            # print reward
            fig1, ax1 = plt.subplots(figsize=(11, 8))
            ax1.plot(range(num_train.value + 1), reward_sum_list)
            ax1.set_title("Episode Reward vs train iteration")
            ax1.set_xlabel("Training iteration")
            ax1.set_ylabel("Episode reward")
            plt.savefig('%s/episode_reward_vs_iteration.png' % (args.save_model_dir))
            plt.clf()

            fig2, ax2 = plt.subplots(figsize=(11, 8))
            ax2.plot(range(num_train.value + 1), mean_reward_list)
            ax2.set_title("Average reward vs train iteration")
            ax2.set_xlabel("Train iteration")
            ax2.set_ylabel("Average reward")
            plt.savefig('%s/average_reward_vs_iteration.png' % (args.save_model_dir))
            plt.clf()

            player.env.end()
            if num_train.value > args.num_train:
                break

            time.sleep(60)

    print("Tester ends ...................")
