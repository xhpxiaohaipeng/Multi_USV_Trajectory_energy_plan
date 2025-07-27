#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import numpy as np
import torch
import argparse
from itertools import chain
os.environ['PARL_BACKEND'] = 'torch'
from env.env_wrappers import VectorEnv
from model.simple_model import MHAModel
from parl.algorithms import MAPPO,QMIX
#from mappo import MAPPO
from model.simple_agent import SimpleAgent
from model.mappo_buffer import SeparatedReplayBuffer
from parl.utils import logger, summary

models = {'MHA':MHAModel}
LR = 1e-3  # learning rate
VALUE_LOSS_COEF = 1  # Value loss coefficient (ie. c_1 in the paper)
ENTROPY_COEF: float = 0.01  # Ejjapntropy coefficient (ie. c_2 in the paper)
HUBER_DELTA = 10.0  # coefficience of huber loss
EPS = 1e-2  # Adam optimizer epsilon (default: 1e-5)
MAX_GRAD_NORM = 10.0  # Max gradient norm for gradient clipping
#EPISODE_LENGTH = 96  # Max length for any episode
GAMMA = 0.99  # discount factor for rewards (default: 0.99)
GAE_LAMBDA = 0.95  # gae lambda parameter (default: 0.95)
LOG_INTERVAL_EPISODES = 1  # time duration between contiunous twice log printing
CLIP_PARAM = 0.2  # ppo clip parameter, suggestion 4 in the paper (default: 0.2)
PPO_EPOCH = 15  # number of epochs for updating using each T data, suggestion 3 in the paper (default: 15)
NUM_MINI_BATCH = 256 # number of batches for ppo, suggestion 3 in the paper (default: 1)
import matplotlib.pyplot as plt
import pandas as pd

def get_act_dim_from_act_space(action_space):
    if action_space.__class__.__name__ == "Discrete":
        act_dim = action_space.n
    else:
        act_dim = action_space.high - action_space.low + 1
    return act_dim

def write_data(data,name,name_model,i=None):
    test = pd.DataFrame(data=data,columns=name)
    if i == None:
        test.to_csv('./results/CSV/{}_{}.csv'.format(name[0],name_model))
    else:
        test.to_csv('./results/CSV/{}_{}_{}.csv'.format(name[0],name_model,i))


def Valid(agents,energy=True,name_model='MHA'):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    env_num = 1
    envs = VectorEnv(args.env_name, env_num, args.seed, random_start=False, test=False, multi_discrete=True,
                     obstacles_dynamic_is=True,energy=energy)
    EPISODE_LENGTH = envs.max_time

    agent_num = len(envs.observation_space)

    #agents = []
    buffers = []
    name_model = name_model
    for agent_id in range(agent_num):
        share_observation_space = envs.share_observation_space[agent_id] if args.use_centralized_V else \
            envs.observation_space[agent_id]
        #
        obs_dim = envs.observation_space[agent_id].shape[0]
        cent_obs_dim = share_observation_space.shape[0]
        #act_dim = get_act_dim_from_act_space(envs.action_space[agent_id])
        # model = models[name_model]
        # model = model(obs_dim, cent_obs_dim, act_dim)
        # algorithm = MAPPO(model, CLIP_PARAM, VALUE_LOSS_COEF, ENTROPY_COEF, LR,
        #                   HUBER_DELTA, EPS, MAX_GRAD_NORM, args.use_popart,
        #                   args.use_value_active_masks)
        # agent = SimpleAgent(algorithm)
        # buffer
        bu = SeparatedReplayBuffer(
            EPISODE_LENGTH, env_num, GAMMA, GAE_LAMBDA, obs_dim, cent_obs_dim,
            envs.action_space[agent_id], args.use_popart)
        #agents.append(agent)
        buffers.append(bu)

    # # restore model
    # for i in range(len(agents)):
    #     model_file = args.model_dir + '/' + args.env_name + '/agent_' + str(i) + '_{}_{}.pt'.format(args.name,
    #                                                                                                 name_model)
    #     if not os.path.exists(model_file):
    #         raise Exception(
    #             'model file {} does not exits'.format(model_file))
    #     agents[i].restore(model_file)

    start = time.time()
    episodes = 1#int(args.train_total_steps)
    rewards_sum_list = []
    for episode in range(episodes):
        obs = envs.reset()
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)
        for agent_id in range(agent_num):
            if not args.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            buffers[agent_id].share_obs[0] = share_obs.copy()
            buffers[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()
        rewards_all = []
        for i in range(env_num):
            rewards_all.append([])
            for j in range(agent_num):
                rewards_all[i].append(0)
        epi = 0
        for step in range(EPISODE_LENGTH):
            # Sample actions
            epi += 1
            values = []
            actions = []
            action_log_probs = []

            for agent_id in range(agent_num):
                value, action, action_log_prob = agents[agent_id].sample(
                    buffers[agent_id].share_obs[step],
                    buffers[agent_id].obs[step],True)
                values.append(value)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions_batch = []
            for env_id in range(actions[0].shape[0]):
                env_actions = []
                for agent_id in range(len(actions)):
                    env_actions.append(actions[agent_id][env_id])
                actions_batch.append(env_actions)

            obs, rewards, dones, infos = envs.step(actions_batch)
            for i, re_threa in enumerate(rewards):
                for j, re in enumerate(re_threa):
                    rewards_all[i][j] += list(rewards[i][j])[0]

            masks = np.ones((env_num, agent_num, 1), dtype=np.float32)

            masks[dones == True] = 0#np.zeros(((dones == True).sum(), 1),dtype=np.float32)
            #print(step,dones)
            if dones.all():
                break
            share_obs = []
            for o in obs:
                share_obs.append(list(chain(*o)))
            share_obs = np.array(share_obs)
            for agent_id in range(agent_num):
                if not args.use_centralized_V:
                    share_obs = np.array(list(obs[:, agent_id]))

                buffers[agent_id].insert(
                    share_obs, np.array(list(obs[:, agent_id])),
                    actions[agent_id], action_log_probs[agent_id],
                    values[agent_id], rewards[:, agent_id], masks[:, agent_id])

        rewards_all = np.array(rewards_all)
        # print(rewards_all)
        rewards_all_mean_threa = np.mean(rewards_all, axis=0)
        # print(rewards_all_mean_threa)
        for i, re in enumerate(rewards_all_mean_threa):
            rewards_all_mean_threa[i] = rewards_all_mean_threa[i] / epi
        # rewards_sum = rewards_sum /self.episode_length/self.n_rollout_threads
        rewards_sum = sum(rewards_all_mean_threa)
        rewards_sum_list.append(rewards_sum)
        # compute return and update network
        with torch.no_grad():
            for agent_id in range(agent_num):
                next_values = agents[agent_id].value(
                    buffers[agent_id].share_obs[-1])
                buffers[agent_id].compute_returns(
                    next_values, agents[agent_id].value_normalizer)
        # log information
        end = time.time()
        agent_rewards = []
        for agent_id in range(agent_num):
            idv_rews = []
            for info in infos:
                if 'individual_reward' in info[agent_id].keys():
                    idv_rews.append(info[agent_id]['individual_reward'])
            individual_rewards = round(np.mean(idv_rews), 3)
            average_episode_rewards = round(
                np.mean(buffers[agent_id].rewards) * EPISODE_LENGTH, 3)
            agent_rewards.append(individual_rewards)
        use_time = round(end - start, 3)

    return np.mean(rewards_sum_list), rewards_all_mean_threa

def train():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    envs = VectorEnv(args.env_name, args.env_num, args.seed,random_start=True,test=False,multi_discrete=True,obstacles_dynamic_is=True,energy=True)
    EPISODE_LENGTH = envs.max_time
    agent_num = len(envs.observation_space)
    env_num = args.env_num

    agents = []
    buffers = []
    name_model = 'MHA'
    for agent_id in range(agent_num):
        share_observation_space = envs.share_observation_space[agent_id] if args.use_centralized_V else \
            envs.observation_space[agent_id]

        obs_dim = envs.observation_space[agent_id].shape[0]
        cent_obs_dim = share_observation_space.shape[0]
        act_dim = get_act_dim_from_act_space(envs.action_space[agent_id])
        #print(act_dim)

        model = models[name_model]
        #print('xx',obs_dim)
        model = model(obs_dim, cent_obs_dim, act_dim)
        #SimpleModel(obs_dim, cent_obs_dim, act_dim)
        algorithm = MAPPO(model, CLIP_PARAM, VALUE_LOSS_COEF, ENTROPY_COEF, LR,
                          HUBER_DELTA, EPS, MAX_GRAD_NORM, args.use_popart,
                          args.use_value_active_masks)
        agent = SimpleAgent(algorithm)
        # buffer
        bu = SeparatedReplayBuffer(
            EPISODE_LENGTH, env_num, GAMMA, GAE_LAMBDA, obs_dim, cent_obs_dim,
            envs.action_space[agent_id], args.use_popart)
        agents.append(agent)
        buffers.append(bu)

    if args.restore:
        # restore model
        for i in range(len(agents)):
            model_file = args.model_dir + '/' + args.env_name + '/agent_' + str(i)+ '_{}_{}.pt'.format(args.name,name_model)
            if not os.path.exists(model_file):
                raise Exception(
                    'model file {} does not exits'.format(model_file))
            agents[i].restore(model_file)

    start = time.time()
    episodes = int(args.train_total_steps) // EPISODE_LENGTH // args.env_num
    rewards_sum_list = []
    reward_sum_valid = []
    max_mean_reward = -np.inf
    max_reward = -np.inf
    for episode in range(episodes):
        obs = envs.reset()
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)
        for agent_id in range(agent_num):
            if not args.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            buffers[agent_id].share_obs[0] = share_obs.copy()
            buffers[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

        rewards_all = []
        for i in range(args.env_num):
            rewards_all.append([])
            for j in range(agent_num):
                rewards_all[i].append(0)
        epi = 0
        for step in range(EPISODE_LENGTH):
            # Sample actions
            epi += 1
            values = []
            actions = []
            action_log_probs = []
            for agent_id in range(agent_num):
                value, action, action_log_prob = agents[agent_id].sample(
                    buffers[agent_id].share_obs[step],
                    buffers[agent_id].obs[step])
                values.append(value)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions_batch = []
            for env_id in range(actions[0].shape[0]):
                env_actions = []
                for agent_id in range(len(actions)):
                    env_actions.append(actions[agent_id][env_id])
                actions_batch.append(env_actions)

            obs, rewards, dones, infos = envs.step(actions_batch)
            for i, re_threa in enumerate(rewards):
                for j, re in enumerate(re_threa):
                    rewards_all[i][j] += list(rewards[i][j])[0]

            masks = np.ones((args.env_num, agent_num, 1), dtype=np.float32)

            masks[dones == True] = 0#np.zeros(((dones == True).sum(), 1),dtype=np.float32)

            share_obs = []
            for o in obs:
                share_obs.append(list(chain(*o)))
            share_obs = np.array(share_obs)
            for agent_id in range(agent_num):
                if not args.use_centralized_V:
                    share_obs = np.array(list(obs[:, agent_id]))

                buffers[agent_id].insert(
                    share_obs, np.array(list(obs[:, agent_id])),
                    actions[agent_id], action_log_probs[agent_id],
                    values[agent_id], rewards[:, agent_id], masks[:, agent_id])
        rewards_all = np.array(rewards_all)
        # print(rewards_all)
        rewards_all_mean_threa = np.mean(rewards_all, axis=0)
        # print(rewards_all_mean_threa)
        for i, re in enumerate(rewards_all_mean_threa):
            rewards_all_mean_threa[i] = rewards_all_mean_threa[i] / epi
        # rewards_sum = rewards_sum /self.episode_length/self.n_rollout_threads
        rewards_sum = sum(rewards_all_mean_threa)
        rewards_sum_list.append(rewards_sum)

        # compute return and update network
        with torch.no_grad():
            for agent_id in range(agent_num):
                next_values = agents[agent_id].value(
                    buffers[agent_id].share_obs[-1])
                buffers[agent_id].compute_returns(
                    next_values, agents[agent_id].value_normalizer)

        # learn
        train_infos = []
        for agent_id in range(agent_num):
            train_info = agents[agent_id].learn(
                buffers[agent_id], PPO_EPOCH, NUM_MINI_BATCH, args.use_popart)
            train_infos.append(train_info)
            buffers[agent_id].after_update()

        # log information
        total_num_steps = (episode + 1) * EPISODE_LENGTH * args.env_num
        if episode % LOG_INTERVAL_EPISODES == 0:
            end = time.time()
            agent_rewards = []
            # valid_reward, valid_reward_each = Valid(agents=agents,energy=True,name_model=name_model)
            # reward_sum_valid.append(valid_reward)
            for agent_id in range(agent_num):
                idv_rews = []
                for info in infos:
                    if 'individual_reward' in info[agent_id].keys():
                        idv_rews.append(info[agent_id]['individual_reward'])
                individual_rewards = round(np.mean(idv_rews), 3)
                average_episode_rewards = round(
                    np.mean(buffers[agent_id].rewards) * EPISODE_LENGTH, 3)
                agent_rewards.append(individual_rewards)
                train_infos[agent_id].update({
                    'individual_rewards':
                        individual_rewards
                })
                train_infos[agent_id].update({
                    "average_episode_rewards":
                        average_episode_rewards
                })

            use_time = round(end - start, 3)
            model_dir = args.model_dir + '/' + args.env_name
            if np.mean(rewards_sum_list) > max_mean_reward:
                print('最大平均验证奖励增加,保存模型参数...')
                # max_mean_reward = np.mean(rewards_sum_list)
                max_mean_reward = np.mean(rewards_sum_list)
                for i in range(len(agents)):
                    model_name = '/agent_' + str(i)+'_max_mean_{}.pt'.format(name_model)
                    agents[i].save(model_dir + model_name)

            if rewards_sum > max_reward:
                # max_reward = rewards_sum
                max_reward = rewards_sum
                print('最大验证奖励增加,最大奖励为：{}, 保存模型参数...'.format(max_reward))
                for i in range(len(agents)):
                    model_name = '/agent_' + str(i)+'_max_{}.pt'.format(name_model)
                    agents[i].save(model_dir + model_name)

            logger.info(
                'Steps: {}, Episodes: {}/{}, Mean episode reward: {}, mean agents rewards {}, Time: {}'
                    .format(total_num_steps, episode, episodes,
                            np.mean(rewards_sum_list), rewards_all_mean_threa, use_time))

            for agent_id in range(agent_num):
                for k, v in train_infos[agent_id].items():
                    agent_k = "agent%i/" % agent_id + k
                    summary.add_scalar(agent_k, v, total_num_steps)
        if episode % 1e4 == 0:
            write_data(rewards_sum_list, ['reward'],name_model,episode/1e4)

            # # save model
            # if not args.restore:
            #     model_dir = args.model_dir + '/' + args.env_name
            #     os.makedirs(os.path.dirname(model_dir), exist_ok=True)
            #     for i in range(len(agents)):
            #         model_name = '/agent_' + str(i)
            #         agents[i].save(model_dir + model_name)
    write_data(rewards_sum_list, ['reward'],name_model)
    plt.plot(rewards_sum_list, label='reward')
    plt.legend()
    plt.savefig('results/reward.png', dpi=300)
    plt.show()
    envs.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_name',
        type=str,
        default='multi_usv',
        help='scenario of MultiAgentEnv')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument(
        '--env_num',
        type=int,
        default= 512,
        help='Number of parallel envs to train')
    parser.add_argument(
        '--train_total_steps',
        type=int,
        default=256e8,
        help='Number of environment steps to train')
    parser.add_argument(
        '--restore',
        action='store_true',
        default=False,
        help='restore or not, must have model_dir')

    parser.add_argument(
        '--name',
        action='store_true',
        default='max',
        help='restore or not, must have model_dir')

    parser.add_argument(
        '--show', action='store_true', default=False, help='display or not')

    parser.add_argument(
        '--model_dir',
        type=str,
        default='results/model',
        help='directory for saving model')

    # Five suggestions mentioned in the paper
    parser.add_argument(
        '--use_popart',
        default=True,
        help=
        'whether to use PopArt to normalize rewards, suggestion 1 in the paper (default: True)'
    )
    parser.add_argument(
        '--use_centralized_V',
        default=True,
        help=
        'whether to use centralized V function, suggestion 2 in the paper (default: True)'
    )
    parser.add_argument(
        "--use_value_active_masks",
        default=True,
        help=
        "whether to mask useless data in value loss, suggestion 5 in the paper (default: True)"
    )

    args = parser.parse_args()
    logger.set_dir('./train_log/' + str(args.env_name))

    train()
