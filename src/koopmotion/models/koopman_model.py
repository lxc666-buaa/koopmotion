'''
Koopman Operator Theoretic model class 
'''

from torch import nn
import torch
import numpy as np
import torch.nn.init as init

'''
Koopman Operator Theoretic model class 
'''


class KoopmanModel(nn.Module):
    def __init__(self, observables, training_args, D=2):
        """
        初始化Koopman模型

        参数:
            observables: 可观测函数（提升函数）
            training_args: 训练参数字典
            D: 状态空间维度，默认为2
        """
        super().__init__()
        self.training_args = training_args

        # 为使用恒等可观测进行重构而附加状态本身的维度
        self.num_rff = self.training_args['num_rff'] + D


        # 注意：我们使用Koopman算子的低秩版本，因为这似乎有助于正则化
        rank = self.training_args['operator_rank']
        # 定义Koopman算子的U和V参数矩阵，用于低秩近似 K = U @ V.T
        self.U = nn.Parameter(torch.empty((self.num_rff, rank), requires_grad=True))
        self.V = nn.Parameter(torch.empty((self.num_rff, rank), requires_grad=True))

        # 使用Xavier均匀初始化初始化U和V参数
        init.xavier_uniform_(self.U)
        init.xavier_uniform_(self.V)

        # 保存可观测函数
        self.observables = observables

    def forward(self, inputs):
        """
        前向传播函数

        参数:
            inputs: 输入数据

        返回:
            lifted_inputs: 提升后的输入
            forward_propagated_lifted_inputs: 在提升空间中前向传播的结果
        """
        # 使用可观测函数将输入提升到高维空间
        lifted_inputs = self.observables(inputs)

        # 在提升空间中应用线性Koopman算子，其中Koopman算子K由self.U @ self.V.T定义
        K = self.U @ self.V.T
        # 在提升空间中前向传播提升后的输入
        forward_propagated_lifted_inputs = K @ lifted_inputs

        return lifted_inputs, forward_propagated_lifted_inputs

