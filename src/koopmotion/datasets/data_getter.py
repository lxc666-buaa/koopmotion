'''
Grabs training data from data folder under data/data_name_selected_in_config/train.npy
'''

import os 
import numpy as np
import matplotlib.pyplot as plt


class TrainingData():
    def __init__(self, training_args, data_args, plotting_args):
        """
        初始化TrainingData对象

        参数:
            training_args: 训练相关参数
            data_args: 数据相关参数，包括系统名称和边界等
            plotting_args: 绘图相关参数
        """
        self.training_args = training_args
        self.data_args = data_args
        self.plotting_args = plotting_args
        # 构建训练数据文件路径
        self.path = os.path.join(os.getcwd(),
                                 'data',
                                 self.data_args['system'],
                                 'train.npy')

        # 获取训练数据的输入和输出
        self.training_inputs, self.training_outputs = self.get_training_data()

    def get_training_data(self):
        '''
        准备轨迹数据用于训练

        返回:
            training_inputs: 训练输入数据
            training_outputs: 训练输出数据
        '''
        # 加载所有数据
        all_data = np.load(self.path)
        print('All data shape', all_data.shape)

        # 获取快照数据（分离成训练输入和输出）
        training_inputs, training_outputs = self.get_one_snapshot_data(all_data)

        # 如果配置要求绘制图像，则绘制训练数据
        if self.plotting_args['plot_figure']:
            self.plot_training_data(training_inputs, training_outputs)

        # 确保数据类型为float32以用于训练
        training_inputs = training_inputs.astype(np.float32)
        training_outputs = training_outputs.astype(np.float32)

        return training_inputs, training_outputs

    def plot_training_data(self, training_inputs, training_outputs):
        '''
        平面轨迹数据绘图器

        参数:
            training_inputs: 训练输入数据
            training_outputs: 训练输出数据
        '''

        # 断言检查，确保输入数据有2个状态（x和y坐标）
        assert training_inputs.shape[
                   0] == 2, 'As of now, plotting is not configured for system with 2 states! Set config file param plot_figure : False'

        # 创建图形
        plt.figure(figsize=(6, 6), dpi=300)

        # 绘制输入和输出数据点
        plt.scatter(*training_inputs, alpha=0.5)  # 输入数据点
        plt.scatter(*training_outputs, alpha=0.5)  # 输出数据点
        plt.title('Training Data')  # 图形标题
        plt.xlabel(self.plotting_args['x_label'])  # x轴标签
        plt.ylabel(self.plotting_args['y_label'])  # y轴标签

        # 计算差值并向量可视化
        difference = training_outputs - training_inputs
        plt.quiver(*training_inputs, difference[0], difference[1])  # 绘制箭头表示状态转移
        plt.xlim(*self.data_args['bounds_x'])  # 设置x轴范围
        plt.ylim(*self.data_args['bounds_y'])  # 设置y轴范围

        # 创建保存路径并保存图像
        folder_path = os.path.join('figures', self.data_args['system'])
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(folder_path + '/training_data.png', bbox_inches='tight')

    def get_one_snapshot_data(self, all_data):
        '''
        通过分离成训练输入和输出来生成快照数据，
        并根据配置中定义的噪声方差添加噪声

        参数:
            all_data: 包含所有数据的numpy数组，形状为(状态数, 样本数, 2)
                     其中第三个维度0表示输入，1表示输出

        返回:
            training_sampled_inputs: 训练输入数据
            training_sampled_outputs: 训练输出数据
        '''

        # 分离输入和输出数据
        training_sampled_inputs = all_data[:, :, 0]
        training_sampled_outputs = all_data[:, :, 1]

        # 添加噪声到数据 - 仅对输出添加噪声以模拟观测噪声，
        # 但如果机器人估计有噪声，也对输入添加噪声
        # training_sampled_inputs  += np.random.normal(0, self.data_args['training_noise_sigma'], training_sampled_inputs.shape)

        # 计算平均向量及其范数
        average_vector = np.mean(training_sampled_inputs - training_sampled_outputs, axis=1)
        average_vector_norm = np.linalg.norm(average_vector)

        # 不对最后一个样本添加噪声，以确保达到目标点
        training_sampled_outputs[:, :-1] += np.random.normal(0, average_vector_norm * self.data_args[
            'training_noise_sigma'], training_sampled_outputs[:, :-1].shape)

        return training_sampled_inputs, training_sampled_outputs

    def normalize_data(self, data):
        """
        归一化数据到[0,1]区间

        参数:
            data: 要归一化的数据，包含x和y坐标

        返回:
            归一化后的数据
        """
        x, y = data

        # 对x坐标进行归一化
        data[0, :] = (x - min(self.data_args['bounds_x'])) / (
                    max(self.data_args['bounds_x']) - min(self.data_args['bounds_x']))
        # 对y坐标进行归一化
        data[1, :] = (y - min(self.data_args['bounds_y'])) / (
                    max(self.data_args['bounds_y']) - min(self.data_args['bounds_y']))

        return data
        
    