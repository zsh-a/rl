import os
from env_full import Environment
import gym
import tianshou as ts

env = Environment('data/u20.txt')

train_envs = Environment('data/u20.txt')
test_envs = Environment('data/u20.txt')

train_envs = ts.env.DummyVectorEnv([lambda: Environment('data/u20.txt',seed=i) for i in range(10)])
test_envs = ts.env.DummyVectorEnv([lambda: Environment('data/u20.txt',seed=i) for i in range(100)])

import torch, numpy as np
from torch import nn

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)
if os.path.exists('dqn.pth'):
    policy.load_state_dict(torch.load('dqn.pth'))
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

policy.eval()
policy.set_eps(0.05)
collector = ts.data.Collector(policy, env, exploration_noise=True)
print(collector.collect(n_episode=1))
# print(train_collector.collect(n_episode=1))

# result = ts.trainer.offpolicy_trainer(
#     policy, train_collector, test_collector,
#     max_epoch=100, step_per_epoch=10000, step_per_collect=10,
#     update_per_step=0.1, episode_per_test=100, batch_size=64,
#     train_fn=lambda epoch, env_step: policy.set_eps(0.1),
#     test_fn=lambda epoch, env_step: policy.set_eps(0.05))

# torch.save(policy.state_dict(), 'dqn.pth')

# print(f'Finished training! Use {result["duration"]}')