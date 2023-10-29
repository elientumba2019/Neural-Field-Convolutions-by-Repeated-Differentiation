import torch
import sys


def train(
        SAVE_PATH,
        args,
        model,
        optim,
        scheduler,
        writer,
        net_dictionary,
        kernel_object,
        monte_carlo_np,
        convolution_fn,
        sampling_fn,
        loss_fn,
        interpolator_fn):
    # ------------------------------------------------------------------------------------------------------------------
    global_iteration = 0
    if args.init_ckpt is not None:
        checkp = torch.load(args.init_ckpt)
        global_iteration = checkp['epoch']

    # ------------------------------------------------------------------------------------------------------------------
    model = model.train()

    # ------------------------------------------------------------------------------------------------------------------
    control_pts_coords, control_pts_vals = kernel_object.get_control_points()
    control_pts_coords = torch.from_numpy(control_pts_coords).cuda().float()
    control_pts_vals = torch.from_numpy(control_pts_vals).cuda().float()
    control_pts_nums = kernel_object.get_n_control_points()

    for step in range(args.num_steps + 1):

        global_iteration += 1
        batch_size = args.batch
        optim.zero_grad()

        if global_iteration % 10000 == 0:
            net_dictionary['ckpt'] = model.state_dict()
            net_dictionary['epoch'] = global_iteration
            net_dictionary['optim'] = optim.state_dict()
            torch.save(net_dictionary, SAVE_PATH + f'/checkpoint_{global_iteration}.pth')

        if global_iteration % 1000 == 0:
            net_dictionary['ckpt'] = model.state_dict()
            net_dictionary['epoch'] = global_iteration
            net_dictionary['optim'] = optim.state_dict()
            torch.save(net_dictionary, SAVE_PATH + f'/current.pth')

        # data sampling
        # ----------------------------------------------------------------------------------------------------------
        input_tensor, monte_carlo_rgb = sampling_fn(args, interpolator_fn, monte_carlo_np)

        # convolution
        # ----------------------------------------------------------------------------------------------------------
        convolution_output = convolution_fn(model,
                                            input_tensor,
                                            control_pts_coords,
                                            control_pts_vals,
                                            control_pts_nums,
                                            args)

        loss = loss_fn(convolution_output.float(), monte_carlo_rgb.float())
        # ----------------------------------------------------------------------------------------------------------

        loss.backward()
        optim.step()

        # ----------------------------------------------------------------------------------------------------------
        if global_iteration % 200 == 0:
            print(f'Iteration : {global_iteration},'
                  f' train loss:, {loss.item()},'
                  f' Batch Size:, {batch_size},'
                  f'kernel : {1 / args.kernel_scale}')

        # ----------------------------------------------------------------------------------------------------------

        writer.add_scalar('Integral Loss', loss.item(), global_iteration)
        scheduler.step()
        sys.stdout.flush()

    net_dictionary['ckpt'] = model.state_dict()
    net_dictionary['epoch'] = global_iteration
    net_dictionary['optim'] = optim.state_dict()
    torch.save(net_dictionary, SAVE_PATH + f'/model_final.pth')
    # ------------------------------------------------------------------------------------------------------------------

