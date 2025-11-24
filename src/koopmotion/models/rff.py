'''
Defines the lifting function, that lifts our states into a space where the dynamics now evolve linearly
'''

import numpy as np
import torch
from torch import nn
import torch.nn.init as init


class LearnedFF(nn.Module):
    '''
    This model lifts its inputs (X) to a higher dimensional space, 
    of dimension (rff_n + states_n, inputs_n). 
    The lifting function is parameterized by by Fourier features,
    defined by learnable weights and biases.

    For each Fourier feature we have a weight or phase (W),
    which is a vector matching the dimension of the number of states of the system
    and a corresponding bias or phase shift (b), which is just one constant.
    '''

    def __init__(self, append_inputs=True, rff_n=100, states_n=2):
        """
        初始化LearnedFF模型，该模型使用可学习的傅里叶特征将输入提升到高维空间

        参数:
            append_inputs (bool): 是否在提升后的特征后追加原始输入状态，默认为True
            rff_n (int): 傅里叶特征的数量，默认为100
            states_n (int): 系统状态的维度数，默认为2
        """
        super().__init__()

        # 存储模型配置参数
        self.append_inputs = append_inputs  # 控制是否追加原始输入
        self.rff_n = rff_n  # 傅里叶特征数量
        self.states_n = states_n  # 状态维度数

        # 创建可学习参数：
        # b: 偏置参数，形状为(rff_n, 1)，每个傅里叶特征对应一个偏置值
        # W: 权重参数，形状为(rff_n, states_n)，每个傅里叶特征对每个状态维度都有一个权重
        self.b = nn.Parameter(torch.empty((self.rff_n, 1), requires_grad=True))
        self.W = nn.Parameter(torch.empty((self.rff_n, states_n), requires_grad=True))

        # 使用Xavier均匀初始化来初始化参数b和W，这种初始化方式在实践中效果较好
        init.xavier_uniform_(self.b)
        init.xavier_uniform_(self.W)

    def forward(self, X):
        """
        前向传播函数，将输入X通过可学习的傅里叶特征映射提升到高维空间

        参数:
            X (Tensor): 输入张量，形状为(states_n, inputs_n)
                   其中states_n是状态维度数，inputs_n是输入样本数

        返回:
            Z (Tensor): 提升后的特征表示，形状取决于append_inputs参数：
                   如果append_inputs=True，形状为(rff_n + states_n, inputs_n)
                   如果append_inputs=False且rff_n>0，形状为(rff_n, inputs_n)
                   如果rff_n=0，直接返回原始输入X
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, device=self.W.device, dtype=self.W.dtype)
        else:
            X = X.to(self.W.device, dtype=self.W.dtype)
        states_n, inputs_n = X.shape

        # 构建偏置项B：
        # 1. 将偏置b从(rff_n, 1)重塑为(rff_n,)然后扩展为(rff_n, inputs_n)
        # 2. 乘以2π得到最终的相位偏移
        self.B = (2 * torch.pi * self.b.reshape(-1, 1)).expand(-1, inputs_n)

        # 计算归一化因子，用于保持特征的能量不变
        norm = 1.0 / torch.sqrt(torch.tensor(float(self.rff_n), device=self.W.device, dtype=self.W.dtype))

        # 如果傅里叶特征数为0，则直接返回原始输入
        if self.rff_n == 0:
            return X
        else:
            sqrt2 = torch.sqrt(torch.tensor(2.0, device=self.W.device, dtype=self.W.dtype))
            Z = norm * sqrt2 * torch.cos((2 * torch.pi * self.W) @ X + self.B)

            # 根据配置决定是否在提升特征后追加原始输入状态
            # 这样做是为了在重构时可以使用恒等可观测函数（identity observable）
            if self.append_inputs:
                Z = torch.concatenate((X, Z), axis=0)

        return Z

