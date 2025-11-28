'''
Defines the loss terms to optimize over in KoopMotion 
'''

import torch
import torch.nn as nn


class LossFunction():
    def __init__(self, loss_type='custom_loss'):
        """
        初始化损失函数类，根据指定的损失类型选择相应的损失计算方法

        参数:
            loss_type (str): 损失函数类型，默认为'custom_loss'
                           可选值: 'mse' (均方误差), 'custom_loss' (自定义损失)
        """
        # 存储损失函数类型
        self.loss_type = loss_type

        # 根据损失类型初始化相应的损失函数
        if self.loss_type == 'mse':
            # 使用PyTorch内置的均方误差损失函数
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == 'custom_loss':
            # 使用自定义的损失函数
            self.loss_fn = custom_loss

    def __call__(self, training_args, lifted_inputs_current, lifted_inputs_next, raw_inputs_current, U, V,
                 lifted_goal_next, kernel,
                 dense_training_points, auxiliary_inputs=None, lifted_auxiliary=None):
        """
        调用损失函数计算损失值

        参数:
            training_args: 训练参数，包含训练过程中的各种配置
            lifted_inputs_current: 当前时刻提升后的输入数据
            lifted_inputs_next: 下一时刻提升后的输入数据
            U: Koopman算子的U矩阵参数
            V: Koopman算子的V矩阵参数
            raw_inputs_current: 当前时刻原始输入（未提升）
            lifted_goal_next: 目标状态提升后的表示
            kernel: 核函数相关参数
            dense_training_points: 密集训练点数据
            auxiliary_inputs: 位于障碍物安全带周围的辅助点（用于避障损失）
            lifted_auxiliary: 提升后的辅助点

        返回:
            根据所选损失函数类型计算得到的损失值
        """
        return self.loss_fn(training_args, lifted_inputs_current, lifted_inputs_next, raw_inputs_current, U, V,
                            lifted_goal_next, kernel,
                            dense_training_points, auxiliary_inputs=auxiliary_inputs,
                            lifted_auxiliary=lifted_auxiliary)


def _get_loss_weight(training_args, key, default=1.0):
    """Safely fetch nested loss weights with backwards compatibility."""

    if isinstance(training_args.get('loss_weights', None), dict):
        if key in training_args['loss_weights']:
            return training_args['loss_weights'][key]
    return training_args.get(key, default)


def custom_loss(training_args, lifted_inputs_current, lifted_inputs_next, raw_inputs_current, U, V,
                lifted_goal_next,
                kernel, dense_training_points, auxiliary_inputs=None, lifted_auxiliary=None):
    """
    自定义损失函数，用于Koopman模型的训练优化

    该损失函数结合了三个不同的损失项：
    1. Koopman线性性损失：衡量提升空间中的线性演化准确性
    2. 目标收敛损失：确保目标状态在Koopman算子作用下的不变性
    3. 散度损失：控制向量场的局部散度特性

    参数:
        training_args: 训练参数字典，包含各种损失权重配置
        lifted_inputs_current: 当前时刻提升后的输入数据，形状为(num_features, batch_size)
        lifted_inputs_next: 下一时刻提升后的输入数据，形状为(num_features, batch_size)
        U: Koopman算子的U矩阵参数，形状为(num_features, rank)
        V: Koopman算子的V矩阵参数，形状为(num_features, rank)
        lifted_goal_next: 目标状态提升后的表示，形状为(num_features, 1)
        kernel: 核函数相关参数，用于散度计算
        dense_training_points: 密集训练点数据，用于散度计算

    返回:
        tuple: (总损失值, 各单项损失组成的字典)
            - 总损失值: 加权后的各项损失之和
            - losses: 包含各项损失的字典，用于消融研究中的权重调整
    """

    # 初始化均方误差损失函数
    mse_loss = nn.MSELoss()
    # 初始化总损失值
    loss = 0

    # 计算Koopman算子K = U @ V.T，这是Koopman算子的低秩近似形式
    K = U @ V.T

    # Koopman线性性损失（带可微掩码）：衡量提升空间中的线性演化准确性
    # 期望下一时刻的提升表示等于当前时刻提升表示经Koopman算子变换后的结果
    predicted_next_lifted = K @ lifted_inputs_current

    # 根据可选障碍物掩码调整对示教数据的信任度
    obstacle_cfg = training_args.get('obstacle', {}) or {}
    obstacle_enabled = obstacle_cfg.get('enabled', False)
    if obstacle_enabled:
        device = lifted_inputs_current.device
        dtype = lifted_inputs_current.dtype
        center = torch.tensor(obstacle_cfg.get('center', [0.0, 0.0]), device=device, dtype=dtype).reshape(2, 1)
        safe_radius = float(obstacle_cfg.get('radius', 0.0)) + float(obstacle_cfg.get('safety_margin', 0.0))
        mask_alpha = float(obstacle_cfg.get('mask_alpha', 10.0))
        distances = torch.linalg.norm(raw_inputs_current - center, dim=0)
        mask_weights = torch.sigmoid(mask_alpha * (distances - safe_radius))
    else:
        mask_weights = torch.ones(lifted_inputs_current.shape[1], device=lifted_inputs_current.device,
                                  dtype=lifted_inputs_current.dtype)

    koopman_error = torch.sum((lifted_inputs_next - predicted_next_lifted) ** 2, dim=0)
    loss_koopman_linearity = torch.mean(mask_weights * koopman_error)

    # 目标收敛损失：确保目标状态在Koopman算子作用下保持不变（即为不动点）
    # 期望目标状态提升表示在Koopman算子作用后仍保持不变
    loss_goal_convergence = mse_loss(lifted_goal_next, K @ lifted_goal_next)

    # 散度损失：控制向量场的局部散度特性
    # 计算密集训练点处向量场的散度值
    divergence = compute_divergence(dense_training_points, kernel, K)
    # 期望散度值为0（即向量场为无散场），计算与0的均方误差
    loss_div_local = mse_loss(
        torch.tensor(0.0, device=divergence.device, dtype=divergence.dtype),
        divergence,
    )

    # 根据配置文件中的权重参数对各项损失进行加权
    divergence_weight = _get_loss_weight(training_args, 'divergence_weight', 1.0)
    koopman_weight = _get_loss_weight(training_args, 'koopman_weight', 1.0)
    convergence_weight = _get_loss_weight(training_args, 'convergence_weight', 1.0)
    repulsion_weight = _get_loss_weight(training_args, 'repulsion_weight', 0.0)

    loss_div_local_weighted = loss_div_local * divergence_weight
    loss_koopman_linearity_weighted = loss_koopman_linearity * koopman_weight
    loss_goal_convergence_weighted = loss_goal_convergence * convergence_weight

    # 避障势场损失：仅在配置启用时生效
    loss_repulsion = torch.tensor(0.0, device=lifted_inputs_current.device, dtype=lifted_inputs_current.dtype)
    if obstacle_enabled and repulsion_weight > 0:
        def repulsion_penalty(predicted_states):
            distances = torch.linalg.norm(predicted_states - center, dim=0)
            return torch.relu(safe_radius - distances) ** 2

        predicted_next_states = predicted_next_lifted[:2, :]
        penalties = [repulsion_penalty(predicted_next_states)]

        if lifted_auxiliary is not None and auxiliary_inputs is not None:
            predicted_aux_next_states = (K @ lifted_auxiliary)[:2, :]
            penalties.append(repulsion_penalty(predicted_aux_next_states))

        loss_repulsion = torch.mean(torch.cat(penalties))

    loss_repulsion_weighted = loss_repulsion * repulsion_weight

    # 计算总损失，为各项加权损失之和
    loss += loss_koopman_linearity_weighted + loss_div_local_weighted + loss_goal_convergence_weighted + loss_repulsion_weighted

    # 返回各项损失组成的字典，用于消融研究中的权重调整分析
    losses = {
        "koopman": loss_koopman_linearity_weighted,  # Koopman线性性损失
        "divergence": loss_div_local_weighted,  # 散度损失
        "goal": loss_goal_convergence_weighted,  # 目标收敛损失
        "repulsion": loss_repulsion_weighted,  # 避障势场损失
    }
    return loss, losses


def compute_divergence(x, kernel, K):
    '''
    计算向量场在演示轨迹点处的散度值

    该函数计算由Koopman算子K和核函数kernel定义的向量场的散度。
    散度衡量向量场在某点处的"发散"程度，对于无散场，散度应为0。

    参数:
        x: 演示轨迹点数据，形状为(num_points, state_dim)
           这些点是向量场尾部的位置
        kernel: 核函数，用于将状态映射到提升空间
        K: Koopman算子矩阵，形状为(num_features, num_features)

    返回:
        torch.Tensor: 所有轨迹点处散度的平均值
    '''
    # 复制输入数据并转置，设置requires_grad=True以支持自动梯度计算
    # x.T的形状变为(state_dim, num_points)
    x = x.clone().detach().T.requires_grad_(True)

    # 计算向量场在提升空间中的表示，然后投影回原始状态空间（取前2个维度）
    # f_x的形状为(num_points, 2)
    f_x = (K @ kernel(x.T))[:2, :].T

    # 初始化散度张量，形状为(num_points,)
    div_f = torch.zeros(x.shape[0], device=x.device)

    # 遍历输出维度（即向量场的每个分量）
    for i in range(f_x.shape[1]):
        # 计算第i个分量对所有输入变量的梯度
        # grad的形状为(num_points, state_dim)
        grad = torch.autograd.grad(f_x[:, i].sum(), x, create_graph=True)[0]

        # 计算散度：对角元素df_i/dx_i的和（迹运算）
        # 这里累加每个分量对其对应输入变量的偏导数
        div_f += grad[:, i]

        # 返回所有轨迹点处散度的平均值
    return torch.mean(div_f)


