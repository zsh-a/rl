#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.fluid.layers.nn import pool2d
import parl
import paddle.nn as nn
import paddle.nn.functional as F


class Model(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        hid1_size = 128
        hid2_size = 128
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)

        self.policy_fc = nn.Linear(in_features=hid2_size, out_features=act_dim)
        self.value_fc = nn.Linear(in_features=hid2_size, out_features=1)

    def policy(self, obs):
        """
        Args:
            obs: A float32 tensor array of shape [B, C, H, W]

        Returns:
            policy_logits: B * ACT_DIM
        """
        h1 = F.relu(self.fc1(obs))
        fc_output = F.relu(self.fc2(h1))
        policy_logits = self.policy_fc(fc_output)

        return policy_logits

    def value(self, obs):
        """
        Args:
            obs: A float32 tensor of shape [B, C, H, W]

        Returns:
            values: B
        """
        h1 = F.relu(self.fc1(obs))
        fc_output = F.relu(self.fc2(h1))
        values = self.value_fc(fc_output)
        values = paddle.squeeze(values, axis=1)
        return values

    def policy_and_value(self, obs):
        """
        Args:
            obs: A tensor array of shape [B, C, H, W]

        Returns:
            policy_logits: B * ACT_DIM
            values: B
        """
        # print(obs)
        h1 = F.relu(self.fc1(obs))
        fc_output = F.relu(self.fc2(h1))
        policy_logits = self.policy_fc(fc_output)
        values = self.value_fc(fc_output)
        # print(policy_logits)
        values = paddle.squeeze(values, axis=1)
        return policy_logits, values
