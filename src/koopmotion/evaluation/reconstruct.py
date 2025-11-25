'''
Reconstructs the learnt vector field 
'''


import numpy as np
import torch 

from koopmotion.models.rff import LearnedFF 
from koopmotion.models.koopman_model import KoopmanModel

import matplotlib.pyplot as plt 

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('font', size=12)


class Reconstructor:
    def __init__(self):
        """
        初始化Reconstructor类
        """
        pass

    def get_flow_estimate(self, training_args, data_args, reconstruction_args, trained_weights_path,
                          comparison_points=None):
        '''
        计算流估计，由初始网格点及其前向传播产生的矢量场定义

        参数:
            training_args: 训练参数
            data_args: 数据参数
            reconstruction_args: 重建参数
            trained_weights_path: 训练权重路径
            comparison_points: 比较点（可选）

        返回:
            point_propagation_prediction: 点传播预测结果
        '''

        # 设置模型
        model = self.setup_model(training_args, data_args, trained_weights_path)

        # 准备初始条件以预测流
        if comparison_points is None:
            # 如果没有提供比较点，则生成均匀网格点
            grid_points = self.get_uniform_grid_points(data_args, reconstruction_args)
            print('Generating uniform grid points \n')
        else:
            # 使用输入的推理点
            grid_points = comparison_points
            print('Using inputted inference points')

        # 准备矢量场矩阵
        num_points = grid_points.shape[1]  # 点的数量
        # 创建用于存储点传播预测的数组
        point_propagation_prediction = np.zeros((data_args['features_n'],
                                                 num_points,
                                                 reconstruction_args['future_state_prediction']))

        # 初始条件或向量的尾部
        point_propagation_prediction[:, :, 0] = grid_points

        # 通过Koopman算子前向传播并"去提升"
        with torch.no_grad():  # 禁用梯度计算以提高效率

            # Koopman算子
            K = model.U @ model.V.T

            # 注意：如果我们在稀疏数据上训练，point_propagation_prediction将根据稀疏数据时间差演变
            for i in range(1, point_propagation_prediction.shape[2]):
                # 计算提升算子
                lifting_operator = self.compute_lifting(point_propagation_prediction[:, :, i - 1],
                                                        model.observables(
                                                            point_propagation_prediction[:, :, i - 1]).numpy())

                # 生成矢量场
                point_propagation_prediction[:, :, i] = self.get_point_propagation_prediction(data_args,
                                                                                              reconstruction_args,
                                                                                              model.observables,
                                                                                              point_propagation_prediction[
                                                                                              :, :, i - 1],
                                                                                              lifting_operator,
                                                                                              K)

        return point_propagation_prediction

    def setup_model(self, training_args, data_args, trained_weights_path):
        '''
        根据训练权重路径加载模型

        参数:
            training_args: 训练参数
            data_args: 数据参数
            trained_weights_path: 训练权重路径

        返回:
            model: 加载的模型
        '''

        # 初始化模型
        lifting_kernel = LearnedFF(training_args['append_inputs'], training_args['num_rff'], data_args['features_n'])
        initialized_model = KoopmanModel(lifting_kernel, training_args, data_args['features_n'])
        optimizer = torch.optim.Adam(initialized_model.parameters(),
                                     lr=float(training_args['lr']),
                                     weight_decay=float(training_args['weight_decay']))

        # 根据权重获取模型
        model, _, _ = self.load_model_weights(training_args, initialized_model, optimizer, trained_weights_path)

        return model

    def get_uniform_grid_points(self, data_args, reconstruction_args):
        '''
        获取用于评估的均匀网格点

        参数:
            data_args: 数据参数
            reconstruction_args: 重建参数

        返回:
            grid_points: 网格点数组，形状为(维度, 点数)
        '''

        # 获取网格行列数
        num_rows = reconstruction_args['num_rows']
        num_cols = reconstruction_args['num_cols']
        num_height = reconstruction_args['num_cols']

        # 根据特征维度生成网格点
        if data_args['features_n'] == 2:
            # 二维情况
            x_values = np.linspace(*data_args['bounds_x'], num_cols)  # x方向网格点
            y_values = np.linspace(*data_args['bounds_y'], num_rows)  # y方向网格点

            x, y = np.meshgrid(x_values, y_values)
            grid_points = np.vstack((x.ravel(), y.ravel()))  # 展平并堆叠

        elif data_args['features_n'] == 3:
            # 三维情况
            x_values = np.linspace(*data_args['bounds_x'], num_cols)
            y_values = np.linspace(*data_args['bounds_y'], num_rows)
            z_values = np.linspace(*data_args['bounds_z'], num_height)

            x, y, z = np.meshgrid(x_values, y_values, z_values)
            grid_points = np.vstack((x.ravel(), y.ravel(), z.ravel()))

        return grid_points  # 形状:(dim, num_points)

    def get_point_propagation_prediction(self, data_args, reconstruction_args, kernel_psi, X, lifting_operator, K):
        '''
        两个版本应该给出相同的输出。列出两个版本用于健全性检查。

        参数:
            data_args: 数据参数
            reconstruction_args: 重建参数
            kernel_psi: 核函数
            X: 当前状态
            lifting_operator: 提升算子
            K: Koopman算子

        返回:
            propagated_system: 传播后的系统状态
        '''

        # 根据估计类型选择不同的方法
        if reconstruction_args['estimate_type'] == 'using_proxy_delifting':
            # 使用代理去提升方法
            propagated_system = np.linalg.pinv(lifting_operator) @ (K @ torch.as_tensor(kernel_psi(X)).numpy())
        elif reconstruction_args['estimate_type'] == 'using_identity_observable':
            # 使用恒等可观测方法
            propagated_system = (K @ torch.as_tensor(kernel_psi(X)).numpy())[:data_args['features_n'], :]

        return propagated_system

    def compute_lifting(self, X, PsiX):
        '''
        计算提升算子，我们使用(不同的)线性去提升方法进行重构

        参数:
            X: 原始数据
            PsiX: 提升后的数据

        返回:
            lifting_operator: 提升算子
        '''

        # 使用最小二乘法计算提升算子
        # PsiX.T (num_points x num_rff), (X.T) (num_pointsxN)
        lifting_operator, _, _, _ = np.linalg.lstsq(PsiX.T, X.T, rcond=None)
        return lifting_operator

    def load_model_weights(self, training_args, model, optimizer, trained_weights_path):
        '''
        加载模型权重

        参数:
            training_args: 训练参数
            model: 模型
            optimizer: 优化器
            trained_weights_path: 训练权重路径

        返回:
            model: 加载权重后的模型
            optimizer: 新的优化器
            lr_scheduler: 学习率调度器
        '''
        print('******Loading model weights for reconstruction from:', trained_weights_path)
        # 加载训练权重
        trained_weights = torch.load(trained_weights_path, weights_only=False)
        model.load_state_dict(trained_weights['model_state_dict'])
        optimizer.load_state_dict(trained_weights['optimizer_state_dict'])
        # 创建学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=1,
                                                       gamma=0.1)
        # 创建新的优化器
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=float(training_args['lr']),
                                     weight_decay=float(training_args['weight_decay']))

        return model, optimizer, lr_scheduler

  