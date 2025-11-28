''' 
Training/optimization functions 
'''

import numpy as np
import yaml as yaml 
import os  
import torch
import matplotlib.pyplot as plt

from koopmotion.datasets.data_getter import TrainingData  

from time import strftime
import sys

# For checking gradients during training 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


class ModelTrainer():
    def __init__(self, training_args, data_args, plotting_args):
        """
        初始化模型训练器

        参数:
            training_args: 训练参数字典，包含训练超参数配置
            data_args: 数据参数字典，包含数据处理相关配置
            plotting_args: 绘图参数字典，包含可视化相关配置
        """
        # 存储各类配置参数
        self.training_args = training_args
        self.data_args = data_args
        self.plotting_args = plotting_args

        # 初始化训练数据处理器
        self.data = TrainingData(self.training_args, self.data_args, self.plotting_args)

    def save_model(self, epoch, model, optimizer, lr_scheduler, loss, config_filename, n_update):
        """
        保存训练好的模型权重和相关状态

        参数:
            epoch: 当前训练轮次
            model: 训练的模型对象
            optimizer: 优化器对象
            lr_scheduler: 学习率调度器对象
            loss: 当前损失值
            config_filename: 配置文件名
            n_update: 更新次数

        返回:
            trained_weights_path: 保存的模型文件路径
        """
        # 构建模型保存路径
        model_path = os.path.join(os.getcwd(), 'trained_weights', self.data_args['system'])
        path_exists = os.path.exists(model_path)
        if not path_exists:
            os.makedirs(model_path)  # 创建目录如果不存在

        # 生成包含时间戳和配置信息的文件名
        time_str = strftime("%Y%m%d-%H%M%S")
        model_filename = time_str + '_' + config_filename + '_' + 'ep' + str(n_update)
        trained_weights_path = os.path.join(model_path, model_filename)

        # 保存模型状态字典和其他训练相关信息
        torch.save({
            'epoch': epoch,  # 训练轮次
            'model_state_dict': model.state_dict(),  # 模型参数
            'training_args_dictionary': model.training_args,  # 训练参数
            'observables': model.observables,  # 可观测量
            'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
            'scheduler_state_dict': lr_scheduler.state_dict(),  # 调度器状态
            'loss': loss,  # 损失值
        }, trained_weights_path)
        print('Saved to:', trained_weights_path, '\n ')

        return trained_weights_path

    def train(self, n_update, model, data, concatenated_data, loss_fn, optimizer, lr_scheduler, config_filename):
        """
        执行模型训练的一个epoch

        参数:
            n_update: 当前更新次数
            model: 要训练的模型
            data: 训练数据元组(input, output)
            concatenated_data: 连接的训练数据
            loss_fn: 损失函数
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            config_filename: 配置文件名

        返回:
            tuple: (保存的模型路径或None, 学习率调度器)
        """
        try:
            # 分离输入和输出数据
            data_input, data_output = data
            training_n = data_output.shape[-1]  # 训练样本数量

            device = data_input.device
            if self.training_args['training_via_trajectories']:
                permutation = torch.arange(training_n, device=device)
            else:
                permutation = torch.randperm(training_n, device=device)

            # 按批次迭代训练
            for batch_i in range(0, training_n, self.training_args['batch_size']):
                optimizer.zero_grad()  # 清零梯度

                # 在一个epoch中，按batch_size分组迭代所有索引
                batch_ints = permutation[batch_i: batch_i + self.training_args['batch_size']]

                # 获取用于散度损失的训练点周围的点
                training_points_for_divergence_loss = self.get_points_around_training(concatenated_data.T)

                # 在障碍物附近采样辅助点，以提供避障梯度
                obstacle_cfg = self.training_args.get('obstacle', {}) or {}
                auxiliary_inputs = None
                lifted_auxiliary = None
                if obstacle_cfg.get('enabled', False):
                    auxiliary_inputs = self.sample_obstacle_points(obstacle_cfg, data_input.device, data_input.dtype)
                    if auxiliary_inputs is not None:
                        lifted_auxiliary, _ = model(auxiliary_inputs)

                # 对当前批次数据进行前向传播
                PsiX, _ = model(data_input[:, batch_ints])  # 当前时刻提升表示
                PsiY, _ = model(data_output[:, batch_ints])  # 下一时刻提升表示
                #K = model.get_K()
                # 假设轨迹的最后一个点是必须收敛的目标点
                K=model.U @ model.V.T
                goal = data_output[:, -1].reshape(-1, 1)
                lifted_goal_next, _ = model(goal)

                # 计算损失
                loss, _ = loss_fn(self.training_args,
                                  PsiX, PsiY,  # 当前和下一时刻提升表示
                                  data_input[:, batch_ints],  # 原始输入用于避障掩码
                                  model.U, model.V,  # Koopman算子参数
                                  lifted_goal_next,  # 提升后的目标点
                                  model.observables,  # 可观测量
                                  training_points_for_divergence_loss,  # 散度计算点
                                  auxiliary_inputs=auxiliary_inputs,  # 障碍物附近采样点
                                  lifted_auxiliary=lifted_auxiliary)  # 提升后的辅助点

                # 反向传播
                loss.backward()

                # 每隔5个epoch记录梯度范数
                log_every_epochs = 5
                if n_update % log_every_epochs == 0 and batch_i == 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
                    writer.add_scalar("grad_norm", total_norm, n_update)

                # 优化步骤
                optimizer.step()
                writer.add_scalar("total", loss, n_update)

                # 打印训练进度信息
            print('lr: ', lr_scheduler.get_last_lr()[0], ' | loss:', loss, ' | epoch:', str(n_update), '/',
                  self.training_args['epochs_n'])

            # 在第0轮和最后一轮保存模型
            if n_update == 0 or n_update == self.training_args['epochs_n'] - 1:
                trained_weights_path = self.save_model(n_update, model, optimizer, lr_scheduler, loss, config_filename,
                                                       n_update)
                lr_scheduler.step()

                return trained_weights_path, lr_scheduler


        except KeyboardInterrupt:
            # 处理键盘中断（Ctrl+C）
            print('KeyboardInterrupt: Training stopped. Do you want to save? [y/n]')
            user_input = sys.stdin.readline().strip()
            if user_input == 'y':
                print('Saving weights.')
                trained_weights_path = self.save_model(self.training_args['epochs_n'], model, optimizer, lr_scheduler,
                                                       loss, config_filename)
                os._exit(0)
            elif user_input == 'n':
                os._exit(0)
            os._exit(0)

        # 学习率调度步进
        lr_scheduler.step()
        return None, lr_scheduler

    def get_points_around_training(self, training_points, num_samples=4):
        '''
        获取训练数据空间点附近的点，用于散度损失计算

        参数:
            training_points: 训练点数据
            num_samples: 采样点数量，默认为4

        返回:
            points: 采样的训练点
        '''
        # 在训练点中随机选择num_samples个索引
        i = torch.randint(0, training_points.shape[0], (num_samples,), device=training_points.device)
        # 获取对应的点并转置
        points = training_points[i].T.contiguous()
        return points

    def sample_obstacle_points(self, obstacle_cfg, device, dtype):
        """
        在障碍物安全带附近均匀采样辅助点，用于避障势场损失。

        参数:
            obstacle_cfg: 障碍物配置（包含圆心、半径和安全余量）
            device: 目标设备
            dtype: 采样点数据类型

        返回:
            Tensor 或 None: 形状为(2, batch_size)的采样点；若未配置采样则返回None
        """

        batch_size = int(obstacle_cfg.get('auxiliary_batch_size', 0))
        if batch_size <= 0:
            return None

        center = torch.tensor(obstacle_cfg.get('center', [0.0, 0.0]), device=device, dtype=dtype).reshape(2, 1)
        radius = float(obstacle_cfg.get('radius', 0.0))
        safety_margin = float(obstacle_cfg.get('safety_margin', 0.0))
        outer_radius = radius + safety_margin

        # 在圆环区域内均匀采样
        angles = 2 * torch.pi * torch.rand(batch_size, device=device, dtype=dtype)
        radii = torch.sqrt(
            torch.rand(batch_size, device=device, dtype=dtype) * (outer_radius ** 2 - radius ** 2) + radius ** 2
        )
        offsets = torch.stack((torch.cos(angles), torch.sin(angles)), dim=0) * radii
        return center + offsets




        
        

