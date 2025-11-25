
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata
import os




class Plotter:
    def __init__(self):
        """
        初始化Plotter类
        """
        pass

    def plot_flow(self, training_args, vector_field):
        '''
        绘制学习到的矢量场

        参数:
            training_args: 训练参数，包含如RFF数量等配置
            vector_field: 矢量场数据，形状为(2, N, 2)
                         - 第一维: x和y坐标
                         - 第二维: 数据点索引
                         - 第三维: 起始点(0)和终点(1)
        '''
        # 跳过间隔，用于稀疏化显示
        skip_n = 5
        # 计算矢量场的变化量（速度分量）
        diff_estimate = vector_field[:, ::skip_n, 1] - vector_field[:, ::skip_n, 0]

        # 关闭所有现有图形
        plt.close('all')

        # 创建新图形
        plt.figure(figsize=(6, 6), dpi=300)
        # 设置图形标题，显示RFF数量
        plt.title('Estimated Vector Field with M = ' + str(training_args['num_rff']))
        # 绘制矢量场箭头图
        plt.quiver(vector_field[0, ::skip_n, 0], vector_field[1, ::skip_n, 0],
                   diff_estimate[0, :], diff_estimate[1, :],
                   color='blue', label='prediction', alpha=0.75)

    def plot_streamlines(self, training_args, data_args, plotting_args, reconstruction_args, vector_field):
        '''
        绘制流函数图

        参数:
            training_args: 训练参数
            data_args: 数据参数，包含边界等信息
            plotting_args: 绘图参数，包含坐标轴标签等
            reconstruction_args: 重建参数，包含网格行列数
            vector_field: 矢量场数据
        '''

        # 关闭所有现有图形
        plt.close('all')

        # 创建新图形
        plt.figure(figsize=(6, 6), dpi=300)
        # 计算矢量场的x和y分量变化量
        u = vector_field[0, :, 1] - vector_field[0, :, 0]  # x方向变化量
        v = vector_field[1, :, 1] - vector_field[1, :, 0]  # y方向变化量

        # 创建规则网格用于插值
        grid_x, grid_y = np.meshgrid(np.linspace(*data_args['bounds_x'], reconstruction_args['num_cols']),
                                     np.linspace(*data_args['bounds_y'], reconstruction_args['num_rows']))

        # 将u和v分量插值到规则网格上
        # vector_field[:, :, 0].T 是起始点坐标
        grid_u = griddata(vector_field[:, :, 0].T, u, (grid_x, grid_y), method='linear')
        grid_v = griddata(vector_field[:, :, 0].T, v, (grid_x, grid_y), method='linear')

        # 绘制流线图，密度为3
        plt.streamplot(grid_x, grid_y, grid_u, grid_v, density=3)
        # 设置图形标题，显示RFF数量
        plt.title('Streamplot ' + str(training_args['num_rff']))
        # 设置坐标轴范围
        plt.xlim(*data_args['bounds_x'])
        plt.ylim(*data_args['bounds_y'])

        # 设置坐标轴标签
        plt.xlabel(plotting_args['x_label'])
        plt.ylabel(plotting_args['y_label'])

        # 创建保存路径并保存图像
        folder_path = os.path.join('figures', data_args['system'])
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(folder_path + '/streamfunction_with_labels.png', bbox_inches='tight')
    
    