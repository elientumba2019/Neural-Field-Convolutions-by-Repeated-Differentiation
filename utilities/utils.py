import os.path
import imageio
import torch
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
from jax._src.third_party.scipy.interpolate import RegularGridInterpolator as RegularGridInterpolatorx
import shutil
from functools import reduce
from .minimal_kernels import minimal_kernel_diracs
from ._kernel import TempKernel2d, Kernel3d, TempKernel1d


def create_or_recreate_folders(folder):
    """
    deletes existing folder if they already exist and
    recreates then. Only valid for training mode. does not work in
    resume mode
    :return:
    """
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        os.mkdir(folder)
    else:
        os.mkdir(folder)


def load_montecarlo_gt(path):
    monte_carlo_ground_truth = np.load(path, allow_pickle=True)
    monte_carlo_ground_truth = monte_carlo_ground_truth.item()['res']
    return monte_carlo_ground_truth


def create_minimal_filter_2d(order, half_size=1.0):
    diracs_x, diracs_y = minimal_kernel_diracs(order, half_size)
    grid = np.stack(np.meshgrid(diracs_x, diracs_x), -1)
    values = np.outer(diracs_y, diracs_y)

    kernel = TempKernel2d()  # Kernel2d()
    kernel.initialize_control_points(grid, values, order)
    return kernel


def create_minimal_kernel_3d(args):
    kernelxs, kernel_ys = minimal_kernel_diracs(0, 1/args.kernel_scale)
    values = reduce(np.multiply.outer, (kernel_ys, kernel_ys, kernel_ys))
    coords = np.stack(np.meshgrid(kernelxs, kernelxs, kernelxs), -1)

    kernel = Kernel3d()
    kernel.initialize_control_points(coords, values)

    return kernel


def create_minimal_filter_1d(order, half_size=1.0):
    diracs_x, diracs_y = minimal_kernel_diracs(order, half_size)
    kernel = TempKernel1d()
    kernel.initialize_control_points(diracs_x, diracs_y, order)
    return kernel


def map_range(values, old_range, new_range):
    NewRange = (new_range[0] - new_range[1])
    OldRange = (old_range[0] - old_range[1])
    new_values = (((values - old_range[0]) * NewRange) / OldRange) + new_range[0]
    return new_values


def build_2d_sampler(x_len, y_len, data, method='linear'):
    x = np.linspace(0, data.shape[0] - 1, x_len)
    y = np.linspace(0, data.shape[1] - 1, y_len)
    return RegularGridInterpolator((y, x), data, method=method)


def build_3d_sampler(x_len, y_len, z_len, data):
    x = np.linspace(-1, 1, x_len)
    y = np.linspace(-1, 1, y_len)
    t = np.linspace(-1, 1, z_len)
    return RegularGridInterpolator((x, y, t), data)


def build_3d_sampler_jax(x_len, y_len, z_len, data):
    x = jnp.linspace(0, data.shape[0] - 1, x_len)
    y = jnp.linspace(0, data.shape[1] - 1, y_len)
    z = jnp.linspace(0, data.shape[2] - 1, z_len)
    return RegularGridInterpolatorx((x, y, z), data, bounds_error=False, fill_value=0.0)


def build_2d_sampler_jax(x_len, y_len, data):
    x = jnp.linspace(0, data.shape[0] - 1, x_len)
    y = jnp.linspace(0, data.shape[1] - 1, y_len)
    return RegularGridInterpolatorx((x, y), data, bounds_error=False, fill_value=0.0)


def build_1d_sampler_jax(x_len, shape, data):
    x = jnp.linspace(0, shape - 1, x_len)
    return RegularGridInterpolatorx((x, ), data, bounds_error=False, fill_value=0.0)