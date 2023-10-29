import torch


def do_2d_conv(model,
               xy_samples,
               ctrl_pts_coords,
               ctrl_vals,
               num_ctrl_pts,
               args):

    output_dims = 3
    samples = xy_samples
    sample_xs = samples[:, :1]
    sample_ys = samples[:, 1:]

    coordinates, values = ctrl_pts_coords, ctrl_vals
    num_ctrl_pts = num_ctrl_pts

    duplicated_xs = torch.repeat_interleave(sample_xs, num_ctrl_pts, dim=1)[..., None]
    duplicated_ys = torch.repeat_interleave(sample_ys, num_ctrl_pts, dim=1)[..., None]

    duplicated_grid = torch.cat([duplicated_xs, duplicated_ys], -1)

    coordinates_reshaped = coordinates[None, ...]
    convolution_coordinates = duplicated_grid + coordinates_reshaped

    # ------------------------------------------------------------------------------------------------------------------

    # sampling the mlp
    integral_values = model(convolution_coordinates)
    diracs = values[None, :, None]
    diracs = torch.repeat_interleave(diracs, output_dims, dim=-1)

    # ------------------------------------------------------------------------------------------------------------------

    convolved_results = (integral_values * diracs).sum(1)
    return convolved_results


def do_3d_conv(model,
               x_grid,
               ctrl_pts_coords,
               ctrl_vals,
               num_ctrl_pts,
               args):

    samples = x_grid
    sample_xs = samples

    coordinates, values = ctrl_pts_coords, ctrl_vals
    num_ctrl_pts = num_ctrl_pts

    duplicated_xs = torch.repeat_interleave(sample_xs[:, None, :], num_ctrl_pts, dim=1)
    coordinates_reshaped = coordinates[None]
    convolution_coordinates = duplicated_xs + coordinates_reshaped

    # ------------------------------------------------------------------------------------------------------------------

    # sampling the mlp
    integral_values = model(convolution_coordinates)
    diracs = values[None, :, None]

    # ------------------------------------------------------------------------------------------------------------------

    convolved_results = (integral_values * diracs).sum(1)
    return convolved_results


def do_video_conv(model,
                  sample_nums_torch,
                  kernel_control_points,
                  kernel_values,
                  n_control_points,
                  args):

    coordinates, values = kernel_control_points, kernel_values  # kernel_object.get_control_points()
    num_ctrl_pts = n_control_points

    samples = sample_nums_torch
    sample_xs = samples[:, :1]
    sample_ys = samples[:, 1:2]
    sample_ts = samples[:, 2:]

    coordinates_reshaped = coordinates[None, :, None]
    duplicated_xs = torch.repeat_interleave(sample_xs, num_ctrl_pts, dim=1)[..., None]
    duplicated_ys = torch.repeat_interleave(sample_ys, num_ctrl_pts, dim=1)[..., None]
    duplicated_ts = torch.repeat_interleave(sample_ts, num_ctrl_pts, dim=1)[..., None]
    duplicated_ts = duplicated_ts + coordinates_reshaped

    duplicated_grid = torch.cat([duplicated_xs, duplicated_ys, duplicated_ts], -1)
    convolution_coordinates_torch = duplicated_grid

    sampled_video = model(convolution_coordinates_torch.view(-1, 3))
    sampled_video = sampled_video.view(-1, num_ctrl_pts, 3)

    diracs = values[None, :, None]

    convolved_results = sampled_video * diracs
    convolved_results = convolved_results.sum(1)

    return convolved_results

