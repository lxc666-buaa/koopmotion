"""
Standalone script to reproduce KoopMotion pipeline for LASA Angle.

Steps:
- Reads config from `configuration_files/lasa_angle_{1,0.01,0.01}/model_config.yaml`
- Generates training data (downsampled and full-res) referencing LASA dataset
- Initializes model components (LearnedFF, KoopmanModel)
- Trains using AdamW and StepLR as in scripts/run_training.py#L50-62
- Reconstructs and plots the learnt vector field and trajectories
- [NEW] Plots and saves eigenvalues of the Koopman operator to results/

Run directly: `python scripts/run_angle_demo.py`
"""

import os
import numpy as np
import yaml as yaml
import torch
import matplotlib.pyplot as plt  # [新增] 导入绘图库

import pyLasaDataset as lasa

from time import strftime

from koopmotion.utils.utils import int_constructor, get_training_time, load_config_args
from koopmotion.training.train import ModelTrainer
from koopmotion.training.loss import LossFunction
from koopmotion.models.rff import LearnedFF
from koopmotion.models.koopman_model import KoopmanModel

from koopmotion.evaluation.reconstruct import Reconstructor
from koopmotion.evaluation.plotter import Plotter
from koopmotion.evaluation.checking_trajectories import run_checking_trajectories


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def plot_eigenvalues(model, system_name):
    """
    [新增] 计算并绘制Koopman算子K的特征值分布与单位圆的关系。
    """
    print("正在计算并绘制特征值...")

    # 获取 U 和 V 并计算 K (确保在CPU上并分离梯度)
    U = model.U.detach().cpu().numpy()
    V = model.V.detach().cpu().numpy()
    K=U@V.T
    #K = model.get_K().detach().cpu().numpy()

    # 计算特征值
    eigvals = np.linalg.eigvals(K)

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 6))

    # 绘制单位圆
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle')

    # 绘制特征值
    ax.scatter(eigvals.real, eigvals.imag, c='red', marker='x', label='Eigenvalues')

    # 设置图形属性
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Eigenvalues of Koopman Operator K ({system_name})')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.legend()

    # 保存图片到 results 文件夹
    save_path = os.path.join('results', system_name, 'eigenvalues.png')
    ensure_dirs([os.path.dirname(save_path)])

    plt.savefig(save_path)
    print(f"特征值图已保存至: {save_path}")
    plt.close()


def generate_lasa_data(pattern_name: str, resolution_training: int, system_name: str):
    """
    Generate LASA training data for a given pattern (e.g., 'Angle' or 'Sine').
    Saves to folders named with `system_name` (e.g., 'lasa_angle_{1,0.01,0.01}', 'lasa_sine').
    """
    data_root = os.path.join(os.getcwd(), 'data', system_name)
    ensure_dirs([data_root])

    def _build_dataset(demos, resolution):
        demos_data = []
        for demo in demos:
            pos = demo.pos  # shape (2, N)
            # 根据分辨率对数据进行降采样
            pos = pos[:, ::resolution]
            # 提取当前时刻的x坐标（除去最后一个点）
            x0 = pos[0, :-1]
            x1 = pos[0, 1:]
            y0 = pos[1, :-1]
            y1 = pos[1, 1:]

            data = np.zeros((2, x0.shape[0], 2), dtype=np.float32)
            data[0, :, 0] = x0
            data[0, :, 1] = x1
            data[1, :, 0] = y0
            data[1, :, 1] = y1
            demos_data.append(data)

        demos_data_all = np.concatenate(demos_data, axis=1)
        return demos_data_all

    # Acquire LASA demos for selected pattern
    dataset = getattr(lasa.DataSet, pattern_name)
    demos = dataset.demos

    # Downsampled training data
    train_ds = _build_dataset(demos, resolution_training)
    np.save(os.path.join(data_root, 'train.npy'), train_ds)

    # Full-resolution data for evaluation overlays
    train_full = _build_dataset(demos, 1)
    np.save(os.path.join(data_root, 'train_res=1.npy'), train_full)

    # Print bounds for config cross-check
    border_dev = 5
    print('Parameters for configuration file data_args bounds_x and bounds_y')
    print('min x', int(np.min(train_ds[0, :, :]) - border_dev), 'max_x', int(np.max(train_ds[0, :, :]) + border_dev))
    print('min y', int(np.min(train_ds[1, :, :]) - border_dev), 'max_y', int(np.max(train_ds[1, :, :]) + border_dev))
    print('train.npy shape', train_ds.shape)


def get_training_data(trainer: ModelTrainer, training_args):
    """
    Mirror of scripts/run_training.py:get_training_data.
    """
    print("Shape of Training data inputs:", trainer.data.training_inputs.shape, '\n')
    inputs = torch.tensor(trainer.data.training_inputs.copy())
    outputs = torch.tensor(trainer.data.training_outputs.copy())

    if training_args['training_on_gpu']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = inputs.to(device)
        outputs = outputs.to(device)

    return [inputs, outputs]


def train_loop(training_args, data_args, plotting_args, config_filename):
    """
    Set up and run training, mirroring scripts/run_training.py behavior.
    """
    # Model
    lifting_kernel = LearnedFF(training_args['append_inputs'], training_args['num_rff'], data_args['features_n'])
    model = KoopmanModel(lifting_kernel, training_args, data_args['features_n'])

    if training_args['training_on_gpu']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

    # Trainer and data
    trainer = ModelTrainer(training_args, data_args, plotting_args)
    training_data_torch = get_training_data(trainer, training_args)
    training_inputs, training_outputs = training_data_torch
    concatenated_training_data_torch = torch.concatenate([training_inputs, training_outputs], axis=1)

    # Train
    time_str_start = strftime("%Y%m%d-%H%M%S")
    print('Date and time before training:', time_str_start)

    trained_weights_path = None
    for n_update in range(training_args['epochs_n']):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_args['lr']),
            weight_decay=float(training_args['weight_decay'])
        )
        loss_fnc = LossFunction(training_args['loss_type'])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            training_args['lr_step_size'],
            gamma=training_args['lr_gamma']
        )

        trained_weights_path_candidate, lr_scheduler = trainer.train(
            n_update,
            model,
            training_data_torch,
            concatenated_training_data_torch,
            loss_fnc,
            optimizer,
            lr_scheduler,
            config_filename,
        )

        if trained_weights_path_candidate is not None:
            trained_weights_path = trained_weights_path_candidate

    time_str_end = strftime("%Y%m%d-%H%M%S")
    print('Date and time after training:', time_str_end)
    time_difference = get_training_time(time_str_start, time_str_end)
    print('Total training time in seconds', time_difference)

    # [修改] 返回值增加了 model
    return trainer, trained_weights_path, model


def reconstruct_and_plot(config_args, trained_weights_path):
    """
    Reconstruct vector field and generate plots for Angle.
    """
    training_args = config_args['training_args']
    data_args = config_args['data_args']
    plotting_args = config_args['plotting_args']
    reconstruction_args = config_args['reconstruction_args']

    reconstructor = Reconstructor()
    vector_field = reconstructor.get_flow_estimate(
        training_args, data_args, reconstruction_args, trained_weights_path
    )

    # Plot flow and streamlines
    plotter = Plotter()
    plotter.plot_flow(training_args, vector_field)
    plotter.plot_streamlines(training_args, data_args, plotting_args, reconstruction_args, vector_field)

    # Save vector field for later use
    if reconstruction_args.get('save_vector_field', False):
        folder_path = os.path.join('results', data_args['system'])
        ensure_dirs([folder_path])
        np.save(os.path.join(folder_path, 'vector_field.npy'), vector_field)

    # Trajectories overlay
    run_checking_trajectories(vector_field, data_args, reconstruction_args, plotting_args)


def main():
    # Choose LASA pattern here: 'Angle' or 'Sine'
    pattern_name = 'Angle'

    # Map to system folder name used by KoopMotion (e.g., 'lasa_angle_{1,0.01,0.01}', 'lasa_sine')
    system_name = 'lasa_' + pattern_name.lower()

    # Load config that corresponds to the chosen pattern/system
    yaml.add_constructor('tag:yaml.org,2002:int', int_constructor, Loader=yaml.SafeLoader)
    config_folder_path = os.path.join(os.getcwd(), 'configuration_files', system_name)
    config_file_path = os.path.join(config_folder_path, 'model_config.yaml')
    config_args = load_config_args(config_file_path)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        config_args['training_args']['training_on_gpu'] = True

    # Generate training data for chosen pattern and system
    generate_lasa_data(pattern_name, config_args['data_args']['resolution_training'], config_args['data_args']['system'])

    # Train
    # [修改] 接收返回的 model
    trainer, trained_weights_path, model = train_loop(
        config_args['training_args'],
        config_args['data_args'],
        config_args['plotting_args'],
        'model_config',
    )

    # [新增] 绘制并保存特征值分布图
    plot_eigenvalues(model, config_args['data_args']['system'])

    # Reconstruct and plot (folders derive from config's data_args.system)
    reconstruct_and_plot(config_args, trained_weights_path)


if __name__ == '__main__':
    main()
