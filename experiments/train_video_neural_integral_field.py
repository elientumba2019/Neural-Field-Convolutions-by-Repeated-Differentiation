import sys
sys.path.append('../')

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import numpy as np
import torch
from utilities import TrainingLog
from model import CoordinateNet_ordinary as CoordinateNet
import decord
from training import train
from utilities import load_montecarlo_gt
from utilities import create_or_recreate_folders
from utilities import create_minimal_filter_1d
from utilities import do_video_conv
from utilities import build_3d_sampler
from utilities import generate_training_samples_video


# torch.manual_seed(200)
# np.random.seed(200)
# random.seed(200)


def _parse_args():
    parser = ArgumentParser("Signal Regression", formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--signal", type=str, help="Signal to use", default='sawtooth')

    parser.add_argument("--results_dir", type=str, help="Output directory", default='outputs/multifr')

    parser.add_argument("--activation", type=str, help="Activation function", default='swish')

    parser.add_argument("--kernel", type=str, help="Kernel Path",
                        default='/home/nnsampi/Desktop/code/My Personnal Code/experiment-playground/Repeated Integration/ckpt grid/gaussian_grid_2d_lerp2_r3.pth')

    parser.add_argument("--num_channels", type=int, default=128, help="Number of channels in the MLP")

    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the MLP")

    parser.add_argument("--num-samples", type=int, default=501, help="Number of samples to use for training.")

    parser.add_argument("--out_channel", type=int, default=3, help="Output Channel number")

    parser.add_argument("--in_channel", type=int, default=2, help="Input Channel number")

    parser.add_argument("--sample-rate", type=int, default=8, help="The rate at which training samples occur.")

    parser.add_argument("--num_plot", type=int, default=48, help="The number of points to plot in the display.")

    parser.add_argument("--max-hidden", type=int, default=10, help="Maximum number of hidden units to display.")

    parser.add_argument("--write_summary", action="store_true", help="Whether to write summary to file")

    parser.add_argument("--rotation", type=bool, default=False, help="Whether to use the Rotation Network")

    parser.add_argument('--learn_rate', type=float, default=1e-3, help="The network's learning rate")

    parser.add_argument('--schedule_step', type=int, default=5000,
                        help="Step at which the learning rate must be decreased")

    parser.add_argument('--schedule_gamma', type=float, default=0.6, help="learning rate decrease factor")

    parser.add_argument("--pe", type=int, default=8, help="number of positional encoding functions")

    parser.add_argument("--resolution", default="1280x720", help="Resolution of the display")

    parser.add_argument("--num-steps", type=int, default=300000, help="Number of training steps.")

    parser.add_argument("--down_coeff", type=int, default=16, help="coefficient by which we down-sample the image ")

    parser.add_argument("--workers", type=int, default=12, help="number of workers")

    parser.add_argument("--fit_type", type=int, default=2, help="Type of network to fit to")

    parser.add_argument("--batch", type=int, default=1024, help="Batch Size For training")

    parser.add_argument("--precision", type=int, default=32, help="Precision of the computation")

    parser.add_argument("--norm_exp", type=int, default=2, help="Normalization exponent")

    parser.add_argument("--experiment_name", type=str, default='', help=" experimemt name")

    parser.add_argument("--norm_layer", type=str, default=None, help="Normalization layer")

    parser.add_argument("--summary", help="summary folder",
                        default='/home/nnsampi/Desktop/code/My Personnal Code/experiment-playground/scripts/temp')

    parser.add_argument("--monte_carlo", type=str, default=None, help="Monte_carlo gt path")

    parser.add_argument("--kernel_scale", type=float, default=0.05, help="Normalization exponent for y")

    parser.add_argument("--init_ckpt", type=str, default=None, help="Initialization checkpoint")

    parser.add_argument("--order", type=int, default=1, help="The polynomial order for the convolution during training")

    return parser.parse_args()


def _main():
    args = _parse_args()
    print(args)

    # ------------------------------------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setting up the type of precision computation to be used during the training
    # ------------------------------------------------------------------------------------------------------------------
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
    create_or_recreate_folders(current_experiment_folder)
    print(f'--------------------- Experiment Name : {experiment_name} ----------------')
    # ------------------------------------------------------------------------------------------------------------------

    SAVE_PATH = current_experiment_folder  # args.summary
    kernel_object = create_minimal_filter_1d(args.order, 1 / args.kernel_scale)
    monte_carlo_gt = load_montecarlo_gt(args.monte_carlo)

    # ------------------------------------------------------------------------------------------------------------------

    if args.fit_type == 1:
        print(f'---------- Fitting First Integral Network ----------')
    else:
        print(f'---------- Fitting Second Integral Network ----------')

    writer = TrainingLog(current_experiment_folder, add_unique_str=False)
    # ------------------------------------------------------------------------------------------------------------------

    # Model creation and creation of a model dictionary which is very useful during the evaluation of the network
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

    # whether we are starting from scratch or staring off from previous checkpoint
    # ------------------------------------------------------------------------------------------------------------------
    if args.init_ckpt != None:
        print(f'------------------------------------ Model loaded with checkpoint from previous training')
        ckpt = torch.load(args.init_ckpt)
        model.load_state_dict(ckpt['ckpt'])
        optim = torch.optim.Adam(model.parameters(), args.learn_rate)  # weight_decay=1e-3d
        optim.load_state_dict(ckpt['optim'])

        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        print(f'------------------------------------ No previous checkpoints were used to load the model')
        optim = torch.optim.Adam(model.parameters(), args.learn_rate)  # weight_decay=1e-3d

    model = model.double() if args.precision == 64 else model.float()
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.schedule_step, gamma=args.schedule_gamma)

    # ------------------------------------------------------------------------------------------------------------------

    sys.stdout.flush()
    loss_function = torch.nn.L1Loss()
    interpolator_fn = build_3d_sampler(monte_carlo_gt.shape[0],
                                       monte_carlo_gt.shape[1],
                                       monte_carlo_gt.shape[2],
                                       monte_carlo_gt)

    print(f'Starting the training randomly : ---------------------')
    train(SAVE_PATH,
          args,
          model,
          optim,
          scheduler,
          writer,
          net_dictionary,
          kernel_object,
          monte_carlo_gt,
          do_video_conv,
          generate_training_samples_video,
          loss_function,
          interpolator_fn)


if __name__ == "__main__":
    _main()
