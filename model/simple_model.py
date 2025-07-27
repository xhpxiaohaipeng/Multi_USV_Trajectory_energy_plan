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

import parl
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import TransformerEncoderLayer
from matplotlib import font_manager
import matplotlib as mpl

font_path = 'simsun.ttc'  # Windows 系统下的微软雅黑字体路径
prop = font_manager.FontProperties(fname=font_path)

import torch
class MHAModel(parl.Model):
    def __init__(self, obs_dim, cent_obs_dim, act_dim):
        super(MHAModel, self).__init__()
        self.act_dim = act_dim

        self.actor = Actor(obs_dim, self.act_dim)
        self.critic = MHA_Critic(cent_obs_dim)

    def policy(self, obs):
        actions = self.actor(obs)
        return actions

    def value(self, cent_obs):
        values = self.critic(cent_obs)
        return values

class Actor(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.multi_discrete = False
        self.ln1 = nn.LayerNorm(obs_dim)
        self.ln2 = nn.LayerNorm(64)
        self.ln3 = nn.LayerNorm(64)
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        if isinstance(act_dim, int):
            self.fc3 = nn.Linear(64, act_dim)
        else:
            self.multi_discrete = True
            self.action_outs = []
            for action_dim in act_dim:
                self.action_outs.append(nn.Linear(64, action_dim))
            self.action_outs = nn.ModuleList(self.action_outs)

    def forward(self, obs):

        x = self.ln1(obs)
        x = F.tanh(self.fc1(x))
        x = self.ln2(x)
        x = F.tanh(self.fc2(x))

        if self.multi_discrete:
            policys = []
            for action_out in self.action_outs:
                policy = action_out(x)
                policys.append(policy)
        else:
            policys = self.fc3(x)
        return policys

class Critic(parl.Model):
    def __init__(self, cent_obs_dim):
        super(Critic, self).__init__()
        self.ln1 = nn.LayerNorm(cent_obs_dim)
        self.ln2 = nn.LayerNorm(64)
        self.ln3 = nn.LayerNorm(64)
        self.fc1 = nn.Linear(cent_obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v_out = nn.Linear(64, 1)

    def forward(self, cent_obs):
        #print(cent_obs.shape)
        x = self.ln1(cent_obs)
        x = F.tanh(self.fc1(x))
        x = self.ln2(x)
        x = F.tanh(self.fc2(x))
        values = self.v_out(x)

        return values

from torch.nn import TransformerEncoder, TransformerEncoderLayer,MultiheadAttention
import numpy as np
import math

class Mutihead_Attention(nn.Module):
    def __init__(self,d_model,dim_k,dim_v,n_heads):
        super(Mutihead_Attention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads

        self.q = nn.Linear(d_model,dim_k)
        self.k = nn.Linear(d_model,dim_k)
        self.v = nn.Linear(d_model,dim_v)

        self.o = nn.Linear(dim_v,d_model)
        self.norm_fact = 1 / math.sqrt(d_model)
        # 定义Dropout层
        self.dropout = nn.Dropout(0.1)
        self.atts = []

    def generate_mask(self,dim):
        # 此处是 sequence mask ，防止 decoder窥视后面时间步的信息。
        # padding mask 在数据输入模型之前完成。
        matirx = np.ones((dim,dim))
        mask = torch.Tensor(np.tril(matirx))

        return mask==1

    def forward(self,x,requires_mask=False):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # 对 x 进行自注意力
        Q = self.q(x).reshape(-1,x.shape[0],x.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        K = self.k(x).reshape(-1,x.shape[0],x.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        V = self.v(x).reshape(-1,x.shape[0],x.shape[1],self.dim_v // self.n_heads) # n_heads * batch_size * seq_len * dim_v

        attention_score = torch.matmul(Q,K.permute(0,1,3,2)) * self.norm_fact
        if requires_mask:
            mask = self.generate_mask(x.shape[1])
            mask = mask.to('cuda')
            attention_score.masked_fill(mask,value=float("-inf")) # 注意这里的小Trick，不需要将Q,K,V 分别MASK,只MASKSoftmax之前的结果就好了

        attention_score = F.softmax(attention_score, dim=-1)
        output = torch.matmul(attention_score,V).reshape(x.shape[0],x.shape[1],-1)


        output = self.o(output)
        return output

LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0
from torch.distributions import Normal
class MHA_Critic(parl.Model):
    def __init__(self, cent_obs_dim,embedding =40,nhead=2, nhid=64, nlayers=1, dropout=0.2):
        super(MHA_Critic, self).__init__()
        self.ln1 = nn.LayerNorm(cent_obs_dim*2)
        self.ln2 = nn.LayerNorm(64)
        self.ln3 = nn.LayerNorm(64)
        self.fc1 = nn.Linear(cent_obs_dim*2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v_out = nn.Linear(64, 1)
        self.MHA =  Mutihead_Attention(30,400,400,nhead)
      
    def forward(self, cent_obs):
        len_shape = len(cent_obs.shape)
        if len_shape == 3:
            cent_obs = torch.reshape(cent_obs,(-1,30*4))
        x_att = torch.reshape(cent_obs,(cent_obs.shape[0],-1,30))
        x_att = self.MHA(x_att)
        x_att = torch.reshape(x_att,(x_att.shape[0],-1))
        x_cocat = torch.concat((cent_obs,x_att),dim=1)

        x = self.ln1(x_cocat)
        x = F.tanh(self.fc1(x))
        x = self.ln2(x)
        x = F.tanh(self.fc2(x))
        values = self.v_out(x)
        if len_shape == 3:
            values = torch.reshape(values,(-1,4,1))

        return values







