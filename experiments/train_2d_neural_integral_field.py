import sys
sys.path.append('../')

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import torch
from utilities import TrainingLog
import imageio
from model import CoordinateNet_ordinary as CoordinateNet
import cv2
from training import train
from utilities import create_or_recreate_folders
from utilities import load_montecarlo_gt
from utilities import create_minimal_filter_2d
import simpleimageio as sio
from utilities import build_2d_sampler
from utilities import do_2d_conv
from utilities import generate_training_samples_2d


# ----------------------------------------------------------------------------------------------------------------------
# torch.set_default_tensor_type(torch.FloatTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float64)
# ----------------------------------------------------------------------------------------------------------------------

def _parse_args():
    parser = ArgumentParser("Signal Regression", formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--activation", type=str, help="Activation function", default='swish')

    parser.add_argument("--num_channels", type=int, default=128, help="Number of channels in the MLP")

    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the MLP")

    parser.add_argument("--out_channel", type=int, default=3, help="Output Channel number")

    parser.add_argument("--in_channel", type=int, default=2, help="Input Channel number")

    parser.add_argument("--rotation", type=bool, default=False, help="Whether to use the Rotation Network")

    parser.add_argument('--learn_rate', type=float, default=1e-3, help="The network's learning rate")

    parser.add_argument('--schedule_step', type=int, default=5000, help="step to decrease the learning rate")

    parser.add_argument('--schedule_gamma', type=float, default=0.6, help="learning rate decrease factor")

    parser.add_argument("--pe", type=int, default=8, help="number of positional encoding functions")

    parser.add_argument("--num-steps", type=int, default=300000, help="Number of training steps.")

    parser.add_argument("--workers", type=int, default=12, help="number of workers")

    parser.add_argument("--batch", type=int, default=1024, help="Batch Size For training")

    parser.add_argument("--precision", type=int, default=32, help="Precision of the computation")

    parser.add_argument("--norm_exp", type=int, default=0, help="Normalization exponent")

    parser.add_argument("--experiment_name", type=str, default='experiment', help="experiment name")

    parser.add_argument("--norm_layer", type=str, default=None, help="Normalization layer")

    parser.add_argument("--summary", help="summary folder", default='')

    parser.add_argument("--monte_carlo", type=str, default=None, help="Monte_carlo gt path")

    parser.add_argument("--kernel_scale", type=float, default=0.05, help="Normalization exponent for y")

    parser.add_argument("--init_ckpt", type=str, default=None, help="Initialization checkpoint")

    parser.add_argument("--dimension", type=int, default=2, help="Whether the filtering is in 2d or in 1d only")

    parser.add_argument("--order", type=int, default=1, help="The polynomial order for the convolution during training")

    parser.add_argument("--seed", type=int, default=100, help="seed value for the training")

    return parser.parse_args()


def _main():
    args = _parse_args()
    print(args)
    # ------------------------------------------------------------------------------------------------------------------

    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    # ------------------------------------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.precision == 32:
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(torch.float32)
        print(f'--------------------- tensor type of computation : {args.precision} ----------------')
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.set_default_dtype(torch.float64)
        print(f'--------------------- tensor type of computation : {args.precision} ----------------')
    # ------------------------------------------------------------------------------------------------------------------

    # create folder where the checkpoints will be saved
    # ------------------------------------------------------------------------------------------------------------------
    experiment_name = args.experiment_name
    current_experiment_folder = os.path.join(args.summary, f'{experiment_name}')

    print(f'--------------------- Experiment Name : {experiment_name} ----------------')

    SAVE_PATH = current_experiment_folder  # args.summary

    # ------------------------------------------------------------------------------------------------------------------
    # kernel control points and montecarlo ground truths loading
    kernel_object = create_minimal_filter_2d(args.order, half_size=1 / args.kernel_scale)

    # loading monte-carlo gts
    monte_carlo_gt = load_montecarlo_gt(args.monte_carlo)

    writer = TrainingLog(current_experiment_folder, add_unique_str=False)

    # creating the model network and the model dictionary
    # ------------------------------------------------------------------------------------------------------------------
    model = CoordinateNet(args.out_channel,
                          args.activation,
                          args.in_channel,
                          args.num_channels,
                          args.num_layers,
                          args.pe,
                          True if args.norm_exp != 0 else False,
                          10,
                          norm_exp=args.norm_exp,
                          norm_layer=args.norm_layer)

    net_dictionary = dict(input=args.in_channel,
                          output=args.out_channel,
                          channels=args.num_channels,
                          layers=args.num_layers,
                          pe=True,
                          encodings=args.pe,
                          normalize_pe=True if args.norm_exp != 0 else False,
                          include_input=True,
                          activation=args.activation)

    if torch.cuda.device_count() > 1:
        print("Total Number of GPUS :", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    # ------------------------------------------------------------------------------------------------------------------

    # loading existing checkpoints if any exist
    # ------------------------------------------------------------------------------------------------------------------
    if args.init_ckpt is not None:
        print(f'------------------------------------ Model loaded with checkpoint from previous training')
        checkpoint = torch.load(args.init_ckpt)
        model.load_state_dict(checkpoint['ckpt'])

        optim = torch.optim.Adam(model.parameters(), args.learn_rate)  # weight_decay=1e-3d
        optim.load_state_dict(checkpoint['optim'])

        # sending to the gpu
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        # manually setting the learning rate
        print('/n --------------- Resetting the learning rate ---------------')
        for g in optim.param_groups:
            g['lr'] = args.learn_rate

    else:
        print(f'------------------------------------ No previous checkpoints were used to load the model')
        create_or_recreate_folders(current_experiment_folder)
        optim = torch.optim.Adam(model.parameters(), args.learn_rate)

    model = model.double() if args.precision == 64 else model.float()
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.schedule_step, gamma=args.schedule_gamma)

    # ------------------------------------------------------------------------------------------------------------------

    # creating the result directory and creating the optimizer for training
    # ------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(args.summary):
        os.makedirs(args.summary)


    # ------------------------------------------------------------------------------------------------------------------
    sys.stdout.flush()

    loss_function = torch.nn.L1Loss()
    interpolator_fn = build_2d_sampler(monte_carlo_gt.shape[0],
                                       monte_carlo_gt.shape[1],
                                       monte_carlo_gt)

    train(
        SAVE_PATH,
        args,
        model,
        optim,
        scheduler,
        writer,
        net_dictionary,
        kernel_object,
        monte_carlo_gt,
        do_2d_conv,
        generate_training_samples_2d,
        loss_function,
        interpolator_fn)


if __name__ == "__main__":
    _main()
    # evaluate()
