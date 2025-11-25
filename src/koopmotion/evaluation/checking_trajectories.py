import numpy as np
import matplotlib.pyplot as plt
import os 

from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata

from functools import partial


# Define ODE function
def get_vector_field(t, state, interp_vx, interp_vy):
    """
    获取给定状态下矢量场的速度分量

    参数:
        t: 时间参数（在此函数中未使用，但可能在其他上下文中需要）
        state: 当前状态，包含x和y坐标 [x, y]
        interp_vx: x方向速度分量的插值函数
        interp_vy: y方向速度分量的插值函数

    返回:
        [vx, vy]: 矢量场在给定点的速度分量列表
                如果插值失败则返回[0, 0]
    """

    # 从状态中解包x和y坐标
    x, y = state

    # 使用插值函数计算给定点的x和y方向速度分量
    vx = interp_vx(x, y)
    vy = interp_vy(x, y)

    # 处理插值失败导致的NaN情况（例如超出边界）
    if np.isnan(vx) or np.isnan(vy):
        return [0, 0]

    # 返回速度矢量
    return [vx, vy]


def plot_streamfunction(vector_field, reconstruction_args, plotting_args):
    """
    绘制矢量场的流线图

    参数:
        vector_field: 矢量场数据，形状为(2, N, 2)，其中:
                      - 第一维: x和y坐标
                      - 第二维: 数据点索引
                      - 第三维: 起始点(0)和终点(1)
        reconstruction_args: 重建参数，包含网格行列数等配置
        plotting_args: 绘图参数，包含坐标轴标签等配置
    """

    # 计算矢量场的x和y分量变化量（速度分量）
    u = vector_field[0, :, 1] - vector_field[0, :, 0]  # x方向变化量
    v = vector_field[1, :, 1] - vector_field[1, :, 0]  # y方向变化量

    # 计算x和y坐标的边界值
    x_min = np.min(vector_field[0, :, 0])  # x坐标最小值
    x_max = np.max(vector_field[0, :, 0])  # x坐标最大值
    y_min = np.min(vector_field[1, :, 0])  # y坐标最小值
    y_max = np.max(vector_field[1, :, 0])  # y坐标最大值

    # 创建规则网格用于插值
    # 注意：这里虽然获取了reconstruction_args中的行列数，但没有实际使用
    reconstruction_args['num_rows'], reconstruction_args['num_cols']

    # 生成规则网格坐标
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, reconstruction_args['num_cols']),
                                 np.linspace(y_min, y_max, reconstruction_args['num_rows']))

    # 将u和v分量插值到规则网格上
    # vector_field[:, :, 0].T 是起始点坐标
    grid_u = griddata(vector_field[:, :, 0].T, u, (grid_x, grid_y), method='linear')
    grid_v = griddata(vector_field[:, :, 0].T, v, (grid_x, grid_y), method='linear')

    # 创建图形并绘制流线图
    plt.figure(figsize=(6, 6), dpi=300)

    # 绘制流线图，密度为2，颜色为灰色
    plt.streamplot(grid_x, grid_y, grid_u, grid_v, density=2, color='gray')
    # 设置坐标轴范围
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # 设置坐标轴标签
    plt.xlabel(plotting_args['x_label'])
    plt.ylabel(plotting_args['y_label'])


def run_checking_trajectories(vector_field, data_args, reconstruction_args, plotting_args):
    """
    运行轨迹检查，绘制流函数和轨迹预测

    参数:
        vector_field: 矢量场数据
        data_args: 数据相关参数
        reconstruction_args: 重建相关参数
        plotting_args: 绘图相关参数
    """

    # 绘制流函数图
    if vector_field.shape[0] == 2:
        plot_streamfunction(vector_field, reconstruction_args, plotting_args)
        plt.axis('off')  # 关闭坐标轴

    # 保存不带标签的流函数图
    folder_path = os.path.join('figures', data_args['system'])
    plt.savefig(folder_path + '/streamfunction_without_labels.png', bbox_inches='tight')

    # 重新绘制带标签的流函数图
    if vector_field.shape[0] == 2:
        plot_streamfunction(vector_field, reconstruction_args, plotting_args)

        # 获取原始未采样的训练数据（将绘制为红色轨迹）
    training_data = np.load(os.path.join(os.getcwd(),
                                         'data',
                                         data_args['system'],
                                         'train_res=1.npy'))  # 这是我们用于原始未采样LASA数据的文件名

    # 提取基础位置和头部位置
    base_positions = vector_field[:, :, 0].T
    head_positions = vector_field[:, :, 1].T

    print('Ground truth demonstration data shape', base_positions.shape)

    # 计算矢量场方向
    vectors = head_positions - base_positions

    # 插值矢量场 - 这只是为了重新组织推理用的向量
    interp_vx = LinearNDInterpolator(base_positions, vectors[:, 0])
    interp_vy = LinearNDInterpolator(base_positions, vectors[:, 1])

    # 获取演示数量和长度
    num_demonstrations = data_args['num_demonstrations']
    demonstration_length = training_data.shape[1] / num_demonstrations
    print('Demonstration length', demonstration_length)

    # 获取时间参数，定义如何在域中前向传播粒子
    res_training = data_args['resolution_training']
    time_steps_actual = demonstration_length
    time_steps_n = time_steps_actual + 50  # 添加额外时间以观察域中其他部分的粒子如何收敛
    end_t = time_steps_n / res_training
    t_span = (0, end_t)

    # 初始化除训练数据初始条件外的一些附加点
    if data_args['multiple_training_trajectories']:
        length_of_training_trajectory = int(training_data.shape[1] / num_demonstrations)
    else:
        length_of_training_trajectory = training_data.shape[1]

    # 计算边界值
    x_min = np.min(vector_field[0, :, 0])
    x_max = np.max(vector_field[0, :, 0])
    y_min = np.min(vector_field[1, :, 0])
    y_max = np.max(vector_field[1, :, 0])

    # 在边界内创建均匀分布的初始条件点
    boundary_delta = 2
    nx, ny = 10, 10
    x_vals = np.linspace(x_min + boundary_delta, x_max - boundary_delta, nx)
    y_vals = np.linspace(y_min + boundary_delta, y_max - boundary_delta, ny)
    initial_condition_list = np.array([[x, y] for x in x_vals for y in y_vals])

    # 初始化用于绘制的轨迹空列表
    x_trajs = []
    y_trajs = []

    # 获取矢量场函数
    vector_field_function = partial(get_vector_field, interp_vx=interp_vx, interp_vy=interp_vy)

    # 对每个初始条件进行迭代
    for i in range(len(initial_condition_list) + num_demonstrations):

        # 使用与演示相同的初始条件，以比较时间上的传播
        if i < num_demonstrations and length_of_training_trajectory * i < training_data.shape[1]:
            y0 = training_data[:, length_of_training_trajectory * i, 0]
        else:
            y0 = initial_condition_list[i - num_demonstrations]

        # 求解常微分方程
        sol = solve_ivp(vector_field_function, t_span, y0, t_eval=np.linspace(0, end_t, int(time_steps_n)))

        # 提取解
        x_traj, y_traj = sol.y
        x_trajs.append(x_traj)
        y_trajs.append(y_traj)

        # 绘制前向传播的轨迹（匹配初始条件的用黑色，否则用蓝色）
        if i < num_demonstrations and length_of_training_trajectory * i < training_data.shape[1]:
            plt.plot(x_traj, y_traj, 'k-', linewidth=3, label='Matching I.C. Predicted Trajectory', zorder=2)
        else:
            plt.plot(x_traj, y_traj, 'b-', label='Non I.C. Predicted Trajectories', alpha=0.5, zorder=1)

            # 绘制初始条件点
        plt.scatter(y0[0], y0[1], color='g', marker='o', label='Initial Conditions (I.C')

        # 绘制最终结束点
        plt.scatter(x_traj[-1], y_traj[-1], color='y', marker='o', alpha=0.7, s=500, label='End', zorder=3)

        # 绘制每条训练轨迹（作为独立段）
        for demo_i in range(num_demonstrations):
            plt.plot(*training_data[:,
                      length_of_training_trajectory * demo_i:length_of_training_trajectory * demo_i + length_of_training_trajectory,
                      0], 'r-', zorder=1)

    # 设置图形属性
    plt.xlabel(plotting_args['x_label'])
    plt.ylabel(plotting_args['y_label'])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # 保存轨迹图
    folder_path = os.path.join('figures', data_args['system'])
    plt.title("Vector Field and Trajectory")
    plt.savefig(folder_path + '/trajectories.png', bbox_inches='tight')