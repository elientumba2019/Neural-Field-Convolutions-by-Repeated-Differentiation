import numpy as np
import torch

from ismael.tools.train_tools import TrainingLog
from ismael.tools.string_tools import print_same_line
from ismael.images.image_io import tev_display_image
from ismael.tools.plot_tools import scatter_plot, rasterize_figure

#=============================================================

def ground_truth_fct(x, mu, sigma):
    
    norm = torch.linalg.vector_norm(x - mu, dim=-1)
    
    # Gaussian
    f1 = torch.exp(-norm**2 / (2 * sigma**2)) * (1. / (sigma * torch.sqrt(torch.tensor(2*np.pi))))


    # circle
    f2 = torch.where(norm < 3, torch.ones_like(x[..., 0]), torch.zeros_like(x[..., 0]))
    
    return f1

#=============================================================

ramp = None

# sum of ramp functions (see Fig. 3 in https://dl.acm.org/doi/pdf/10.1145/15922.15921 for 1D version)
def ramp_fct(order, x, dirac_pos, dirac_val):
    condition = torch.logical_and(x[...,0:1] >= dirac_pos[:,0], x[...,1:2] >= dirac_pos[:,1])
    if order == 0:
        global ramp # this procedure avoids that the constant is created in every iteration
        if ramp is None:
            ramp = torch.ones((x.shape[0], dirac_val.shape[0])).cuda()
    elif order == 1:
        ramp = (x[..., 0:1] - dirac_pos[:,0]) * (x[..., 1:2] - dirac_pos[:,1])
    else:
        assert False, "Higher-order polynomial kernels not yet implemented"
    ramps = torch.where(condition, ramp, torch.zeros_like(ramp))
    #return torch.einsum('k,ijk->ij', dirac_val, ramps)   
    return torch.einsum('k,ik->i', dirac_val, ramps)   

#=============================================================

# a regular 2D grid of coordinates
def regular_grid_2D(interval, sample_count_per_dim):
    x_1d = torch.linspace(interval[0], interval[1], steps=sample_count_per_dim)
    return torch.stack(torch.meshgrid(x_1d, x_1d, indexing='xy'), dim=-1).cuda()

#=============================================================

# convert pytorch tensor to numpy
def download_tensor(t):
    return t.detach().cpu().numpy()

#=============================================================

def main():

    # params
    dirac_count = 21                 # the number of Diracs per dimension
    poly_order = 1                  # the order of the polynomial (0: constant, 1: linear, 2: quadratic, ...)
    batch_size = 256 * 256              # how many samples in each training iteration
    training_interval = [-1.5, 5]     # interval on which to sample during training
    dirac_interval = [-1,1]         # interval on which the Diracs should be initialized
    sum_loss_weight = 0.1         # weight of the loss that enforces that Diracs sum to 1
    training_iter = 20000           # number of training iterations
    learning_rate = 1 * 1e-4       # "learning rate" (no learning here, actually...)
    lr_decay_factor = 1 #0.9        # learning rate decay factor
    lr_decay_freq = 2000            # in which intervals should the learning rate decay factor be applied
    # enable_merging = True         # merge Diracs if they are too close together
    # merge_thres = 0.05            # threshold determining merging distance
    test_resolution = 512           # test resolution
    
    pruning_iter = training_iter - 2000     # in which iteration should close-to-zero Diracs be pruned
    pruning_thres = [4 * 1e-2, 1e-3]        # threshold for pruning (differetn one for each order)
    
    #-------------------------------------
    
    # Dirac positions and values to optimize
    dirac_x = torch.linspace(dirac_interval[0], dirac_interval[1], dirac_count).cuda().requires_grad_(True)
    dirac_y = torch.linspace(dirac_interval[0], dirac_interval[1], dirac_count).cuda().requires_grad_(True)

    dirac_val = torch.zeros((dirac_count * dirac_count,)).cuda().requires_grad_(True)
    dirac_active = torch.ones_like(dirac_val)

    #----------------------------

    # set up optimizer
    training_log = TrainingLog(log_dir="./log/")
    optim = torch.optim.Adam(lr=learning_rate, params=[dirac_val]) # dirac_x, dirac_y,
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=lr_decay_factor)
    
    # run optimization
    for iter in range(training_iter):
        
        # sample positions
        x = torch.rand((batch_size, 2)).cuda() * (training_interval[1] - training_interval[0]) + training_interval[0]

        dirac_pos = torch.stack(torch.meshgrid(dirac_x, dirac_y, indexing='xy'), dim=-1).reshape(dirac_count**2, 2)

        # produce kernel
        kernel = ramp_fct(poly_order, x, dirac_pos, dirac_val)
        
        # sample ground truth
        g = ground_truth_fct(x, 0, 1 / 3.0)
        
        # losses
        recon_loss = torch.mean(torch.abs(kernel - g))
        #recon_loss = torch.mean(torch.abs(kernel - g))
        sum_loss = torch.abs(torch.sum(dirac_val)) # enforce that Diracs sum to one
        # loss = recon_loss + sum_loss_weight * sum_loss
        loss = recon_loss
        
        # run Adam
        optim.zero_grad()
        loss.backward()
        optim.step()

        # learning rate decay
        if (iter + 1) % lr_decay_freq == 0:
            lr_scheduler.step()

        with torch.no_grad():
            dirac_val *= dirac_active

        # if iter == pruning_iter:
        #     with torch.no_grad():
        #         dirac_active = torch.where(
        #             torch.abs(dirac_val) > pruning_thres[poly_order],
        #             torch.ones_like(dirac_val),
        #             torch.zeros_like(dirac_val))

        # telemetry
        if iter % 20 == 0:
            training_log.add_scalar("recon_loss", recon_loss, iter)
            training_log.add_scalar("sum_loss", sum_loss, iter)
            training_log.add_scalar("total_loss", loss, iter)

        if iter % 200 == 0:        
            print_same_line(f"{iter} -- {download_tensor(loss)}")
            
    training_log.close()
    print("\n")

    #----------------------------

    global ramp
    ramp = None

    # convert to numpy
    dirac_val_np = download_tensor(dirac_val)
    dirac_x_np = download_tensor(dirac_x)
    dirac_y_np = download_tensor(dirac_y)
    dirac_active_np = download_tensor(dirac_active)

    print(dirac_val)
    print(dirac_x)
    print(dirac_y)

    x_c, y_c = torch.meshgrid(dirac_x, dirac_y)
    sample_grid = torch.cat([x_c[..., None] ,y_c[..., None]], -1)




    save_dictionary = {
        'ckpt': dirac_val.view(dirac_count, dirac_count)[None, None],
        'radius': 1,
        'n_ctrl_pts': dirac_count ** 2,
        'ctrl_pts': sample_grid
    }

    torch.save(save_dictionary, '/home/nnsampi/Desktop/code/My Personnal Code/experiment-playground/Repeated Integration/1D fourier feature/kernels/2d/gauss_th.pth')

    # print()
    # print(dirac_val_np.shape)
    # print(dirac_x_np.shape)
    # print(dirac_y_np.shape)
    # print(dirac_x_np)
    # print(dirac_y_np)
    # exit()


    print(f"vals: {dirac_val_np}")
    #print(f"pos_x: {dirac_x_np}")
    #print(f"pos_y: {dirac_y_np}")

    dirac_val_np = dirac_val_np[dirac_active_np == 1]
    dirac_pos_np = download_tensor(dirac_pos)[dirac_active_np == 1]
    # THE ABOVE TWO LINES ARE THE FINAL RESULTS

    # producing high-resolution evaluation image
    test_samples = regular_grid_2D(training_interval, test_resolution).reshape((test_resolution**2, 2))
    test_kernel = ramp_fct(poly_order, test_samples, dirac_pos, dirac_val).reshape((test_resolution, test_resolution))
    test_kernel_np = download_tensor(test_kernel)

    g_np = download_tensor(ground_truth_fct(test_samples, 0, 1/3.0).reshape((test_resolution, test_resolution)))

    tev_display_image("gt", g_np[..., None])
    tev_display_image("opt", test_kernel_np[..., None])

    # display Dirac positions
    scatter_fig = scatter_plot(dirac_pos_np, point_size=20, ranges=training_interval[1], size=(6,6), caption="Diracs")
    scatter_rast = rasterize_figure(scatter_fig)
    tev_display_image("diracs", scatter_rast)

    print(f"#diracs: {dirac_pos_np.shape[0]}")

    # see how well the Diracs sum to 1
    #dirac_y_sum = np.sum(dirac_y_np)
    #print(f"sum: {dirac_y_sum}")

#=============================================================    

if __name__ == "__main__":
    main()
    print("\n=== TERMINATED ===")