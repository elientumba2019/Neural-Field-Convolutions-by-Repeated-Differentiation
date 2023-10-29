import os.path

import imageio
import matplotlib.pyplot as plt
import torch
import numpy as np
from utilities import minimal_kernel_diracs, TempKernel2d, TempKernel3d, TempKernel1d
from model import CoordinateNet_ordinary as CoordinateNet
from utilities import do_2d_conv, do_3d_conv, do_video_conv
import sys
from functools import reduce
import click
from utilities import save_mesh
from utilities import create_or_recreate_folders


def save_frames(frames, path):
    for i in range(frames.shape[2]):

        if i % 10 == 0:
            print(f'saved : {i}')

        if i < 10:
            filename = f'000{i}.png'
        elif 10 <= i < 100:
            filename = f'00{i}.png'
        elif 100 <= i < 1000:
            filename = f'0{i}.png'
        else:
            filename = i

        to_save = (np.clip(frames[..., i, :], 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(path, filename), to_save)


def create_minimal_kernel_1d(order, half_size=1.0):
    diracs_x, diracs_y = minimal_kernel_diracs(order, half_size)
    kernel = TempKernel1d()
    kernel.initialize_control_points(diracs_x, diracs_y, order)
    return kernel


def load_diracs(path, scale):
    diracs = torch.load(path)
    vals = diracs['ckpt']
    coords = diracs['ctrl_pts']

    values = np.outer(vals, vals)
    coords = np.stack(np.meshgrid(coords, coords), -1)

    kernel_object = TempKernel2d()
    kernel_object.initialize_control_points(coords, values, 1)
    kernel_object.shrink_kernel(scale)

    return kernel_object


def create_minimal_kernel_3d(size):
    kernel_xs, kernel_ys = minimal_kernel_diracs(0, size)
    values = reduce(np.multiply.outer, (kernel_ys, kernel_ys, kernel_ys))

    coord_vals = np.stack(np.meshgrid(kernel_xs, kernel_xs, kernel_xs), -1)
    kernel = TempKernel3d()
    kernel.initialize_control_points(coord_vals, values, 0)

    return kernel


def load_network(net_path,
                 shape,
                 precision=32,
                 modality=1):
    weights = torch.load(net_path)
    model = CoordinateNet(weights['output'],
                          weights['activation'],
                          weights['input'],
                          weights['channels'],
                          weights['layers'],
                          weights['encodings'],
                          weights['normalize_pe'],
                          10,
                          norm_exp=2).cuda()

    # load the weights into the network
    model.load_state_dict(weights['ckpt'])
    model = model.eval()
    model = model.double() if precision == 64 else model.float()
    # ------------------------------------------------------------------------------------------------------------------

    # generate coordinates to sample
    if modality == 1:
        coords_x = np.linspace(-1, 1, shape[0], endpoint=True)
        coords_y = np.linspace(-1, 1, shape[1], endpoint=True)
        xy_grid = np.stack(np.meshgrid(coords_x, coords_y, indexing='ij'), -1)
        convolution_tensor = np.zeros((xy_grid.shape[0], xy_grid.shape[1], 3))
    else:
        coords_x = np.linspace(-1, 1, shape[0], endpoint=True)
        coords_y = np.linspace(-1, 1, shape[1], endpoint=True)
        coords_z = np.linspace(-1, 1, shape[2], endpoint=True)
        xy_grid = np.stack(np.meshgrid(coords_x, coords_y, coords_z, indexing='ij'), -1)

        if modality == 2:
            convolution_tensor = np.zeros((xy_grid.shape[0], xy_grid.shape[1], xy_grid.shape[2], 1))
        else:
            convolution_tensor = np.zeros((xy_grid.shape[0], xy_grid.shape[1], xy_grid.shape[2], 3))

    xy_grid = torch.from_numpy(xy_grid).float().contiguous().cuda()
    xy_grid = xy_grid.double() if precision == 64 else xy_grid.float()

    return model, xy_grid.float(), np.float32(convolution_tensor)


def create_minimal_filter_2d(order, half_size=1.0):
    diracs_x, diracs_y = minimal_kernel_diracs(order, half_size)
    grid = np.stack(np.meshgrid(diracs_x, diracs_x), -1)
    values = np.outer(diracs_y, diracs_y)

    kernel = TempKernel2d()  # Kernel2d()
    kernel.initialize_control_points(grid, values, order)
    return kernel


def evaluate(kern_path,
             net_path,
             shape,
             kernel_scale,
             conv_fn,
             precision=32,
             block_size=32,
             modality=1):
    if precision == 32:
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.set_default_dtype(torch.float64)

    # ------------------------------------------------------------------------------------------------------------------

    if modality == 1:
        kernel_object = load_diracs(kern_path, kernel_scale)
    elif modality == 2:
        kernel_object = create_minimal_kernel_3d(1 / kernel_scale)
    else:
        kernel_object = create_minimal_kernel_1d(1, 1 / kernel_scale)

    model, xy_grid_torch, convolution_tensor = load_network(net_path, shape, 32, modality)
    model = model.eval()

    # ------------------------------------------------------------------------------------------------------------------

    control_pts_coords, control_pts_vals = kernel_object.get_control_points()
    control_pts_coords = torch.from_numpy(control_pts_coords).cuda().float()
    control_pts_vals = torch.from_numpy(control_pts_vals).cuda().float()
    control_pts_nums = kernel_object.get_n_control_points()

    # ------------------------------------------------------------------------------------------------------------------
    sub_block = block_size

    with torch.no_grad():

        for i in range(0, xy_grid_torch.shape[0], sub_block):
            print(f'major row : {i}')
            sys.stdout.flush()

            for j in range(0, xy_grid_torch.shape[1], sub_block):
                current_grid = xy_grid_torch[i:i + sub_block, j:j + sub_block]

                if modality > 1:
                    for k in range(0, xy_grid_torch.shape[2], sub_block):
                        current_grid = xy_grid_torch[i:i + sub_block, j:j + sub_block, k:k + sub_block]
                        cH, cW, cD, _ = current_grid.shape
                        current_output = conv_fn(model,
                                                 current_grid.contiguous().view(-1, 3),
                                                 control_pts_coords,
                                                 control_pts_vals,
                                                 control_pts_nums,
                                                 None)

                        current_output = current_output.view(cH, cW, cD, convolution_tensor.shape[-1])
                        convolution_tensor[i:i + sub_block, j:j + sub_block,
                        k:k + sub_block] = current_output.cpu().numpy()

                else:
                    cH, cW, _ = current_grid.shape
                    current_output = conv_fn(model,
                                             current_grid.contiguous().view(-1, 2),
                                             control_pts_coords,
                                             control_pts_vals,
                                             control_pts_nums,
                                             None)

                    current_output = current_output.view(cH, cW, 3)
                    convolution_tensor[i:i + sub_block, j:j + sub_block] = current_output.cpu().numpy()

    # ------------------------------------------------------------------------------------------------------------------
    return convolution_tensor


@click.command()
@click.option("--model_path", default='', help="path to data")
@click.option("--kernel_path", default='', help="path to kernel")
@click.option("--save_path", default='', help="path to save output")
@click.option("--modality", default=1, help="modality flag")
@click.option("--width", default=128, help="signal width")
@click.option("--height", default=128, help="signal height")
@click.option("--depth", default=100, help="signal depth (3d and video)")
@click.option("--block_size", default=32, help="sub block size")
@click.option("--kernel_scale", default=20, help="sub block size")
def run_evaluation(model_path,
                   kernel_path,
                   save_path,
                   modality,
                   width,
                   height,
                   depth,
                   block_size,
                   kernel_scale):

    # create the folder where to save the evaluated model
    create_or_recreate_folders(save_path)

    path = model_path
    kern_path = kernel_path

    conv_fn = do_2d_conv
    if modality == 2:
        conv_fn = do_3d_conv
    elif modality == 3:
        conv_fn = do_video_conv

    output_tensor = evaluate(kern_path,
                             path,
                             (width, height, depth),
                             kernel_scale,
                             conv_fn,
                             modality=modality,
                             block_size=block_size)

    # images
    if modality == 1:
        to_save = (np.clip(output_tensor, 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(save_path, 'output.png'), to_save)

    # geometry
    if modality == 2:
        save_mesh(output_tensor, save_path, 'mesh.ply')

    # Videos
    elif modality == 3:
        save_frames(output_tensor, save_path)


if __name__ == '__main__':
    run_evaluation()
