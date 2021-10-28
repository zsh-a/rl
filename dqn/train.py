#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


from math import inf
import os
import gym
import numpy as np
import parl
from parl.core import paddle
from parl.utils import logger, ReplayMemory


from model import Model
from agent import Agent
from parl.algorithms import DQN
import sys 
# sys.path.append("..") 
from env_full import Environment
import paddle


LEARN_FREQ = 5  # training frequency
MEMORY_SIZE = 200000
MEMORY_WARMUP_SIZE = 2000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3 #0.0005
GAMMA = 0.9#0.99

# @parl.remote_class
# class Main(object):


# train an episode
def run_train_episode(agent, env, rpm):
    total_reward = 0
    obs = env.reset()
    # print(obs)
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)
        # if env.can_charge_next(action) is False:
        #     action = 0 
        next_obs, reward, done, _ = env.step(action)
        rpm.append(obs, action, reward, next_obs, done)


        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs,
            batch_done) = rpm.sample_batch(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                    batch_next_obs, batch_done)


        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward



# evaluate 5 episodes
def run_evaluate_episodes(agent, env, eval_episodes=5, render=False):
    eval_reward = []
    times_run = []
    for i in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        times = 0
        trace = []
        while True:
            action = agent.predict(obs)
            # if env.can_charge_next(action) is False:
            #     action = 0
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
            # if len(trace) == 0 or trace[-1] != action:
            trace.append(action)
            times += info["time"]
        eval_reward.append(episode_reward)
        times_run.append(times)
    env.print_info()
    print(trace)
    return np.mean(eval_reward),np.mean(times_run)



def main():
    env = Environment('../data/u20.txt')

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))
    np.random.seed(0)
    paddle.seed(0)
    # set action_shape = 0 while in discrete control environment
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0)


    # build an agent
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)

    agent = Agent(
        alg, act_dim=act_dim, e_greed=0.1, e_greed_decrement=1e-6)


    save_path = './model_21.ckpt'
    if os.path.exists(save_path):
        agent.restore(save_path)


    # warmup memory
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)


    max_episode = 800000



    # start training
    episode = 0
    while episode < max_episode:
        # train part
        for i in range(50):
            total_reward = run_train_episode(agent, env, rpm)
            episode += 1


        # test part
        eval_reward,times = run_evaluate_episodes(agent, env, render=False,eval_episodes=1)
        logger.info('episode:{} e_greed:{} Test reward:{} run times:{}'.format(
            episode, agent.e_greed, eval_reward,times))

        if episode % 10000 == 0:
            # save the parameters to ./model.ckpt
            agent.save(save_path)



if __name__ == '__main__':
    # parl.connect("localhost:6006",distributed_files=['data/u20.txt','model.ckpt'])
    # obj = Main()
    # obj.main()
    main()