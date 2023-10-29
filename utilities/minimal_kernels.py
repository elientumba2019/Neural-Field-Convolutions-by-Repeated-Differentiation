import numpy as np
import matplotlib.pyplot as plt
# from ismael.tools.plot_tools import rasterize_figure
# from ismael.images.image_io import send_to_tev
import torch
import jax.numpy as jnp
import jax

# @jax.jit
def min0(s):
    return lambda x: jnp.where(
        jnp.abs(x) < s,
        jnp.ones_like(x) / (2 * s),
        jnp.zeros_like(x))

# @jax.jit
def min1(s):
    return lambda x: jnp.where(
        jnp.abs(x) < s,
        jnp.where(
            x < 0,
            (x + s) / s ** 2,
            (-x + s) / s ** 2,
        ),
        jnp.zeros_like(x))

# @jax.jit
def min2(s):
    basic_fct = lambda x: jnp.where(
        jnp.abs(x) <= 3,
        jnp.where(
            jnp.abs(x) <= 1,
            3 - x ** 2,
            jnp.where(
                x < -1,
                0.5 * (3 + x) ** 2,
                0.5 * (-3 + x) ** 2,
            ),
        ),
        jnp.zeros_like(x))
    return lambda x: basic_fct(x * 3 / s) * 3 / s / 8


# =============================================================

# Fig. 1 https://dl.acm.org/doi/pdf/10.1145/15922.15921
def minimal_kernel(order, s):
    if order == 0:  # box
        return lambda x: np.where(
            np.abs(x) < s,
            np.ones_like(x) / (2 * s),
            np.zeros_like(x))

    elif order == 1:  # tent
        return lambda x: np.where(
            np.abs(x) < s,
            np.where(
                x < 0,
                (x + s) / s ** 2,
                (-x + s) / s ** 2,
            ),
            np.zeros_like(x))

    elif order == 2:  # quadratic b-spline
        basic_fct = lambda x: np.where(
            np.abs(x) <= 3,
            np.where(
                np.abs(x) <= 1,
                3 - x ** 2,
                np.where(
                    x < -1,
                    0.5 * (3 + x) ** 2,
                    0.5 * (-3 + x) ** 2,
                ),
            ),
            np.zeros_like(x))

        return lambda x: basic_fct(x * 3 / s) * 3 / s / 8
    else:
        assert False, "Only orders 0-2 implemented"


# =============================================================

# the Diracs corresponding to the kernel above
def minimal_kernel_diracs(order, s):
    if order == 0:
        dirac_x = np.array([-s, s])
        dirac_y = np.array([1., -1.]) * 0.5 / s
    elif order == 1:
        dirac_x = np.array([-s, 0., s])
        dirac_y = np.array([1., -2., 1.]) * 0.5 * 2 / s ** 2  # used to be * 4 / s
    elif order == 2:
        dirac_x = np.array([-s, -s / 3., s / 3., s])
        dirac_y = np.array([1., -3., 3., -1.]) * 0.5 * 3 / s
    else:
        assert False, "Only orders 0-2 implemented"

    return dirac_x, dirac_y


def minimal_kernel_diracs2(order, s):
    if order == 0:
        dirac_x = np.array([-s, s])
        dirac_y = np.array([1., -1.]) / (dirac_x[1] - dirac_x[0])
    elif order == 1:
        dirac_x = np.array([-s, 0., s])
        dirac_y = np.array([1., -2., 1.]) / ((dirac_x[1] - dirac_x[0]) ** 2)
    elif order == 2:
        dirac_x = np.array([-s, -s / 3., s / 3., s])
        dirac_y = np.array([1., -3., 3., -1.]) / ((dirac_x[1] - dirac_x[0]) ** 3)  # * 0.5 * 27 / s
    else:
        assert False, "Only orders 0-2 implemented"

    return dirac_x, dirac_y


# =============================================================

def main():
    s = 1.

    # ------------------------------------

    viz_interval = [-s, s]
    viz_samples = 7

    fct_0 = minimal_kernel(0, s)
    fct_1 = minimal_kernel(1, s)
    fct_2 = minimal_kernel(2, s)

    x = np.linspace(viz_interval[0], viz_interval[1], viz_samples)
    y0 = fct_0(x)
    y1 = fct_1(x)
    y2 = fct_2(x)

    print(y1)
    print(x)

    # code added by Elie to save kernel --------------------------------------------------------------------------------
    optimizable_grid = torch.from_numpy(y1[None])
    default_sampling_grid_torch = torch.from_numpy(x)

    save_dictionary = {
        'ckpt': optimizable_grid,
        'radius': s,
        'n_ctrl_pts': viz_samples,
        'ctrl_pts': default_sampling_grid_torch
    }
    save_path = '/home/nnsampi/Desktop/code/My Personnal Code/experiment-playground/Repeated Integration/1D fourier feature/kernels/minimal kernels/MB'
    torch.save(save_dictionary, save_path + f'/linear_r{s}_w_ctrl_nctrl_{viz_samples}.pth')
    # ------------------------------------------------------------------------------------------------------------------

    # check if area under the curve is one
    print(f"area under curve 0: {np.mean(y0) * 2 * s}")
    print(f"area under curve 1: {np.mean(y1) * 2 * s}")
    print(f"area under curve 2: {np.mean(y2) * 2 * s}")

    # sanity check: put discrete Diracs and build (repeated) SAT
    # -----------------------
    def set_diracs_and_integrate(order):

        def dirac_pos_to_index(p):
            return ((p + s) / (2 * s) * (viz_samples - 1)).astype(np.int32)

        # draw Diracs
        y_dir = np.zeros_like(x)
        dirac_x, dirac_y = minimal_kernel_diracs(order, s)
        dirac_x_idx = dirac_pos_to_index(dirac_x)
        for idx, val in zip(dirac_x_idx, dirac_y):
            y_dir[idx] = val

        # repeated SAT
        y_int = y_dir
        for _ in range(order + 1):
            y_int = np.cumsum(y_int)
        y_int /= viz_samples ** order

        return y_dir, y_int

    # -------------------------

    y_dir_0, y_int_0 = set_diracs_and_integrate(0)
    y_dir_1, y_int_1 = set_diracs_and_integrate(1)
    y_dir_2, y_int_2 = set_diracs_and_integrate(2)

    # visualize everything
    fig, ax = plt.subplots(3, figsize=(10, 10))
    ax[0].plot(x, y0)
    ax[0].plot(x, y1)
    ax[0].plot(x, y2)
    ax[0].set_title("Kernels")

    ax[1].plot(x, y_dir_0)
    ax[1].plot(x, y_dir_1)
    ax[1].plot(x, y_dir_2)
    ax[1].set_yscale('symlog')
    ax[1].set_title("Diracs")

    ax[2].plot(x, y_int_0)
    ax[2].plot(x, y_int_1)
    ax[2].plot(x, y_int_2)
    ax[2].set_title("Integrated Diracs")

    plt.tight_layout()
    # fig_rast = rasterize_figure(fig)
    # send_to_tev("minimal kernel", fig_rast)


# =============================================================

if __name__ == "__main__":
    main()
    print("\n=== TERMINATED ===")
