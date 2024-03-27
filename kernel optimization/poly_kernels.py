import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from ismael.tools.train_tools import TrainingLog
from ismael.tools.string_tools import print_same_line

#=============================================================

def ground_truth_fct(x, mu, sigma):
    
    # Gaussian (for blurring)
    f1 = 1. / (sigma * np.sqrt(2*math.pi)) * np.exp(-(x - mu)*(x - mu) / (2 * sigma**2))
    
    # derivative of Gaussian (for edge detection)
    f2 = -x / sigma**2 * np.exp(-(x - mu)*(x - mu) / (2 * sigma**2))

    return f1

#=============================================================

# a ramp function (see Fig. 3 in https://dl.acm.org/doi/pdf/10.1145/15922.15921)
def ramp_fct(order, shift):
    return lambda x: torch.where(x >= shift, (x-shift)**order / np.math.factorial(order), torch.zeros_like(shift))

def ramp_fct_np(order, shift):
    return lambda x: np.where(x >= shift, (x-shift)**order / np.math.factorial(order), np.zeros_like(shift))

#=============================================================

# check if Diracs are too close to each other and merge them in that case
def merge_diracs(dirac_x, dirac_y, dirac_active, merge_thres):

    with torch.no_grad():

        dirac_count = len(dirac_x)

        dirac_y *= dirac_active

        # iterate over all Diracs
        for idx in range(1, dirac_count):
            curr_dirac_x = dirac_x[idx]
            
            # find previous active Dirac
            prev_index = idx - 1
            while not dirac_active[prev_index] == 1.:
                prev_index -= 1
            prev_dirac_x = dirac_x[prev_index]
            
            # get distance
            diff = torch.abs(curr_dirac_x - prev_dirac_x)
            
            # merge Diracs
            if diff < merge_thres:
                dirac_y[idx] = dirac_y[idx] + dirac_y[prev_index]
                dirac_y[prev_index] = 0
                dirac_x[prev_index] = 10000 * (dirac_x[prev_index] + 10) # put away
                dirac_active[idx-1] = 0
                print("\nMERGE")

#=============================================================

def main():

    # params
    dirac_count = 13                 # the number of Diracs
    poly_order = 1                  # the order of the polynomial (0: constant, 1: linear, 2: quadratic, ...)
    batch_size = 1000               # how many samples in each training iteration 
    training_interval = [-2, 2]     # interval on which to sample during training
    eval_interval = [-2, 2]       # interval on which to evaluate the fit
    dirac_interval = [-1,1]         # interval on which the Diracs should be initialized
    sum_loss_weight = 0.1           # weight of the loss that enforces that Diracs sum to 1
    training_iter = 30000           # number of training iterations
    learning_rate = 8 * 1e-4        # "learning rate" (no learning here, actually...)
    lr_decay_factor = 0.5           # learning rate decay factor
    lr_decay_freq = 10000           # in which intervals should the learning rate decay factor be applied
    enable_merging = False           # merge Diracs if they are too close together
    merge_thres = 0.005              # threshold determining merging distance
    
    #-------------------------------------

    #TODO: run on the entire thing on the GPU to gain some performance

    # training points (could/should be replaced with random ones)
    x = torch.linspace(training_interval[0], training_interval[1], batch_size)
    
    # evaluation of the ground truth function
    g = ground_truth_fct(x, 0, 1 / 3)

    # Dirac positions and values to optimize
    dirac_x = torch.linspace(dirac_interval[0], dirac_interval[1], dirac_count).requires_grad_(True)
    dirac_y = torch.zeros_like(dirac_x).requires_grad_(True)
    dirac_active = torch.ones_like(dirac_x) # keep track of active Diracs (merged ones become inactive)

    # put together the ramps
    ramps = []
    for idx in range(len(dirac_x)):
        ramps.append(ramp_fct(poly_order, dirac_x[idx]))
        
    #----------------------------

    # set up optimizer
    training_log = TrainingLog(log_dir="./log/")
    optim = torch.optim.Adam(lr=learning_rate, params=[dirac_y, dirac_x])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=lr_decay_factor)
    
    # run optimization
    for iter in range(training_iter):
        
        # produce kernel
        kernel = torch.zeros_like(x)
        for ramp, factor in zip(ramps, dirac_y):
            kernel += factor * ramp(x)
        
        # losses
        recon_loss = torch.mean(torch.square(kernel - g))
        sum_loss = torch.abs(torch.sum(dirac_y)) # enforce that Diracs sum to one
        loss = recon_loss + sum_loss_weight * sum_loss
        
        # run Adam
        optim.zero_grad()
        loss.backward()
        optim.step()

        # learning rate decay
        if (iter + 1) % lr_decay_freq == 0:
            lr_scheduler.step()

        # telemetry
        if iter % 20 == 0:
            training_log.add_scalar("recon_loss", recon_loss, iter)
            training_log.add_scalar("sum_loss", sum_loss, iter)
            training_log.add_scalar("total_loss", loss, iter)

        if iter % 200 == 0:
            
            print_same_line(f"{iter} -- {loss.detach().numpy()}")
            
            # merge Diracs that are too close together
            if enable_merging:
                merge_diracs(dirac_x, dirac_y, dirac_active, merge_thres)


    training_log.close()
    print("\n")

    #----------------------------
                   
    # convert to numpy
    kernel_np = kernel.detach().numpy()
    dirac_x_np = dirac_x.detach().numpy()
    dirac_y_np = dirac_y.detach().numpy()
    dirac_active_np = dirac_active.detach().numpy()

    print(f"pos: {dirac_x_np}")
    print(f"vals: {dirac_y_np}")
    print(f"active: {dirac_active_np}")

    # delete inactive Diracs
    dirac_x_np = dirac_x_np[dirac_active_np == 1]
    dirac_y_np = dirac_y_np[dirac_active_np == 1]
    # THE ABOVE TWO LINES ARE THE FINAL RESULTS

    save_dictionary = {
        'ckpt': dirac_y_np,
        'radius': 1,
        'n_ctrl_pts': dirac_count,
        'ctrl_pts': dirac_x_np
    }

    torch.save(save_dictionary, '/home/nnsampi/Desktop/code/My Personnal Code/experiment-playground/Repeated Integration/1D fourier feature/kernels/1d/gaussian')

    print(f'dirac x np: {dirac_y_np}')
    print(f'dirac y np : {dirac_y_np}')




    # se how well the Diracs sum to 1
    dirac_y_sum = np.sum(dirac_y_np)
    print(f"sum: {dirac_y_sum}")

    # larger evaluation (see how it behaves outside of the trained range; does it matter?)
    test_samples = np.linspace(eval_interval[0], eval_interval[1], 100000)
    test_eval = np.zeros_like(test_samples)
    for dir_x, dir_y in zip(dirac_x_np, dirac_y_np):
        test_eval += dir_y * ramp_fct_np(poly_order, dir_x)(test_samples)
    print(f"last value: {test_eval[-1]}")

    # plot everything
    fig, ax = plt.subplots(3)
    ax[0].plot(x, g)
    ax[0].plot(x, kernel_np)
    ax[1].stem(dirac_x_np, dirac_y_np)
    ax[2].plot(test_samples, test_eval)

    ax[0].set_title("Kernel")
    ax[1].set_title("Diracs")
    ax[2].set_title("Larger eval")
    
    ax[0].set_xlim(training_interval)
    ax[1].set_xlim(training_interval)
    ax[2].set_xlim(eval_interval)

    plt.tight_layout()
    plt.show()

#=============================================================    

if __name__ == "__main__":
    main()
    print("\n=== TERMINATED ===")