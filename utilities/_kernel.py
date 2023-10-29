import numpy as np
from abc import ABC
# from .utils import map_range, bilinear_interpolate_numpy, bilinear_interpolate_scipy
from scipy import signal
import torch
# from .minimal_kernels import minimal_kernel_diracs
from functools import reduce


class Kernel1d(ABC):
    def __init__(self, xs=None, ys=None):
        self.original_control_points = {}
        self.current_control_points = {}

        self.original_size = 0
        self.current_size = self.original_size
        self.min = 0
        self.max = 0

        if xs is not None and ys is not None:
            self.min = xs.min()
            self.max = xs.max()
            self.original_size = 1
            self.initialize_control_points(xs, ys)

    def initialize_control_points(self, xs, ys):
        self.min = xs.min()
        self.max = xs.max()
        print(xs.dtype)
        self.original_size = 1
        self.current_size = self.original_size

        for i in range(len(xs)):
            self.original_control_points[xs[i]] = ys[i]

        self.current_control_points = self.original_control_points

    def stretch_kernel(self, factor):
        items = self.current_control_points.items()
        self.current_size = self.current_size * factor
        self.current_control_points = {}

        for i in items:
            self.current_control_points[i[0] * factor] = i[1] / factor ** 2

    def shrink_kernel(self, factor):
        items = self.current_control_points.items()
        self.current_size = self.current_size / factor
        self.current_control_points = {}

        for i in items:
            self.current_control_points[i[0] / factor] = i[1] * (factor ** 2)

    def get_min(self):
        return self.min()

    def get_max(self):
        return self.max

    def get_original_size(self):
        return self.original_size

    def get_current_size(self):
        return self.current_size

    def get_n_control_points(self):
        return len(self.original_control_points.items())

    def get_control_points(self):
        xs = list(self.current_control_points.keys())
        ys = []
        for x in xs:
            ys.append(self.current_control_points[x])

        return np.array(xs), np.array(ys)

    def get_original_control_points(self):
        xs = list(self.original_control_points.keys())
        ys = []
        for x in xs:
            ys.append(self.original_control_points[x])

        return np.array(xs), np.array(ys)

    def __str__(self):
        return f'Original: {str(self.original_control_points)} \ncurrent : {self.current_control_points}'

    def reset_kernel_scale(self):
        self.current_control_points = self.original_control_points
        self.current_size = self.original_size


class TempKernel1d(ABC):
    def __init__(self, xs=None, ys=None):
        self.original_control_points = {}
        self.current_control_points = {}

        self.original_size = 0
        self.current_size = self.original_size
        self.min = 0
        self.max = 0
        self.order = None

        if xs is not None and ys is not None:
            self.min = xs.min()
            self.max = xs.max()
            self.original_size = 1
            self.initialize_control_points(xs, ys)

    def initialize_control_points(self, xs, ys, order):
        self.min = xs.min()
        self.max = xs.max()

        self.original_size = 1
        self.current_size = self.original_size
        self.order = order

        for i in range(len(xs)):
            self.original_control_points[xs[i]] = ys[i]

        self.current_control_points = self.original_control_points

    def stretch_kernel(self, factor):
        items = self.current_control_points.items()
        self.current_size = self.current_size * factor
        self.current_control_points = {}

        for i in items:
            self.current_control_points[i[0] * factor] = i[1] / factor ** 2

    def shrink_kernel(self, factor):
        items = self.current_control_points.items()
        self.current_size = self.current_size / factor
        self.current_control_points = {}

        for i in items:
            self.current_control_points[i[0] / factor] = i[1] / ((1 / factor) ** (self.order + 1))

    def get_min(self):
        return self.min()

    def get_max(self):
        return self.max

    def get_original_size(self):
        return self.original_size

    def get_current_size(self):
        return self.current_size

    def get_n_control_points(self):
        return len(self.original_control_points.items())

    def get_control_points(self):
        xs = list(self.current_control_points.keys())
        ys = []
        for x in xs:
            ys.append(self.current_control_points[x])

        return np.array(xs), np.array(ys)

    def get_original_control_points(self):
        xs = list(self.original_control_points.keys())
        ys = []
        for x in xs:
            ys.append(self.original_control_points[x])

        return np.array(xs), np.array(ys)

    def __str__(self):
        return f'Original: {str(self.original_control_points)} \ncurrent : {self.current_control_points}'

    def reset_kernel_scale(self):
        self.current_control_points = self.original_control_points
        self.current_size = self.original_size


# ----------------------------------------------------------------------------------------------------------------------

class Kernel2d(ABC):
    def __init__(self, input_coordinates=None, zs=None):
        self.original_control_points = {}
        self.current_control_points = {}

        self.original_size = 0
        self.current_size = self.original_size

        self.min_x = 0.0
        self.max_x = 0.0
        self.min_y = 0.0
        self.max_y = 0.0

        if input_coordinates is not None and zs is not None:
            self.original_size = 1
            self.initialize_control_points(input_coordinates, zs)

    def initialize_control_points(self, input_coordinates, zs):

        # print(input_coordinates.shape)
        # print(zs.shape)

        H, W, C = input_coordinates.shape

        self.min_x = input_coordinates[:, :, 0].min()
        self.max_x = input_coordinates[:, :, 0].max()

        self.min_y = input_coordinates[:, :, 1].min()
        self.max_y = input_coordinates[:, :, 1].max()

        self.original_size = 1
        self.current_size = self.original_size

        for i in range(H):
            for j in range(W):
                current_x = input_coordinates[i, j]
                current_y = zs[i, j]

                self.original_control_points[tuple(current_x.tolist())] = current_y

        self.current_control_points = self.original_control_points

    def stretch_kernel(self, factor):
        # self.reset_kernel_scale()

        items = self.current_control_points.items()
        self.current_size = self.current_size * factor
        self.current_control_points = {}

        for i in items:
            current_coord = i[0]
            current_y = i[1]
            self.current_control_points[(current_coord[0] * factor, current_coord[1] * factor)] = (
                        current_y / (factor ** 2) ** 2)

    def x_stretch(self, factor, exponent=2):
        # self.reset_kernel_scale()

        items = self.current_control_points.items()
        self.current_size = self.current_size * factor
        self.current_control_points = {}

        for i in items:
            current_coord = i[0]
            current_y = i[1]
            self.current_control_points[(current_coord[0] * factor, current_coord[1] * 1)] = (
                        current_y / (factor ** exponent) ** 1)

    def y_stretch(self, factor, exponent=1):
        # self.reset_kernel_scale()

        items = self.current_control_points.items()
        self.current_size = self.current_size * factor
        self.current_control_points = {}

        for i in items:
            current_coord = i[0]
            current_y = i[1]
            self.current_control_points[(current_coord[0] * 1, current_coord[1] * factor)] = (
                        current_y / (factor ** exponent) ** 1)

    def rotate(self, angle):
        items = self.current_control_points.items()
        self.current_control_points = {}

        for i in items:
            current_coord = i[0]
            nx = current_coord[0] * np.cos(angle) - current_coord[1] * np.sin(angle)
            ny = current_coord[0] * np.sin(angle) + current_coord[1] * np.cos(angle)
            current_y = i[1]
            self.current_control_points[(nx, ny)] = current_y

    def shrink_kernel(self, factor):
        # self.reset_kernel_scale()

        items = self.current_control_points.items()
        self.current_size = self.current_size / factor
        self.current_control_points = {}

        for i in items:
            current_coord = i[0]
            current_y = i[1]
            self.current_control_points[(current_coord[0] / factor, current_coord[1] / factor)] = current_y * (
                        factor ** 2 ** 2)

    def reset_kernel_scale(self):
        self.current_control_points = self.original_control_points
        self.current_size = self.original_size

    def get_min(self):
        return self.min_x, self.min_y

    def get_max(self):
        return self.max_x, self.max_y

    def get_original_size(self):
        return self.original_size

    def get_current_size(self):
        return self.current_size

    def get_n_control_points(self):
        return len(self.original_control_points.items())

    def get_control_points(self):
        # get original_control points
        xs = self.current_control_points.keys()
        current_size = self.get_current_size()

        # print(xs)
        x_array = []
        y_array = []
        z_array = []

        for x in xs:
            current_x, current_y = x
            # print(f'{current_x} --- {current_y}')
            # current_x = current_x * (1 / current_size)
            # current_y = current_y * (1 / current_size)
            # print(f'{current_x} --- {current_y}')

            current_z = self.current_control_points[x]
            x_array.append(current_x)
            y_array.append(current_y)
            z_array.append(current_z)

        x_array = np.array(x_array)
        y_array = np.array(y_array)
        z_array = np.array(z_array)

        coordinates_array = np.concatenate([x_array[:, None], y_array[:, None]], -1)

        return coordinates_array, z_array

    def get_original_control_points(self):
        # get original_control points
        xs = self.original_control_points.keys()
        current_size = self.get_current_size()

        # print(xs)
        x_array = []
        y_array = []
        z_array = []

        for x in xs:
            current_x, current_y = x
            # print(f'{current_x} --- {current_y}')
            # current_x = current_x * (1 / current_size)
            # current_y = current_y * (1 / current_size)
            # print(f'{current_x} --- {current_y}')

            current_z = self.original_control_points[x]
            x_array.append(current_x)
            y_array.append(current_y)
            z_array.append(current_z)

        x_array = np.array(x_array)
        y_array = np.array(y_array)
        z_array = np.array(z_array)

        coordinates_array = np.concatenate([x_array[:, None], y_array[:, None]], -1)

        return coordinates_array, z_array

    def __str__(self):
        return f'Original: {str(self.original_control_points)} \ncurrent : {self.current_control_points}'


class TempKernel2d(ABC):
    def __init__(self, input_coordinates=None, zs=None):
        self.original_control_points = {}
        self.current_control_points = {}

        self.original_size = 0
        self.current_size = self.original_size
        self.order = None

        self.min_x = 0.0
        self.max_x = 0.0
        self.min_y = 0.0
        self.max_y = 0.0

        if input_coordinates is not None and zs is not None:
            self.original_size = 1
            self.initialize_control_points(input_coordinates, zs)

    def initialize_control_points(self, input_coordinates, zs, order):

        # print(input_coordinates.shape)
        # print(zs.shape)

        H, W, C = input_coordinates.shape

        self.min_x = input_coordinates[:, :, 0].min()
        self.max_x = input_coordinates[:, :, 0].max()

        self.min_y = input_coordinates[:, :, 1].min()
        self.max_y = input_coordinates[:, :, 1].max()

        self.original_size = 1
        self.current_size = self.original_size
        self.order = order

        for i in range(H):
            for j in range(W):
                current_x = input_coordinates[i, j]
                current_y = zs[i, j]

                self.original_control_points[tuple(current_x.tolist())] = current_y

        self.current_control_points = self.original_control_points

    def initialize_control_points2(self, input_coordinates, zs, order):

        # print(input_coordinates.shape)
        # exit()

        self.min_x = input_coordinates[..., 0].min()
        self.max_x = input_coordinates[..., 0].max()

        self.min_y = input_coordinates[..., 1].min()
        self.max_y = input_coordinates[..., 1].max()

        self.original_size = 1
        self.current_size = self.original_size
        self.order = order

        for i in range(len(input_coordinates)):
            current_x = input_coordinates[i]
            current_y = zs[i]
            self.original_control_points[tuple(current_x.tolist())] = current_y

        self.current_control_points = self.original_control_points

    def stretch_kernel(self, factor):
        # self.reset_kernel_scale()

        items = self.current_control_points.items()
        self.current_size = self.current_size * factor
        self.current_control_points = {}

        for i in items:
            current_coord = i[0]
            current_y = i[1]
            self.current_control_points[(current_coord[0] * factor, current_coord[1] * factor)] = (
                        current_y / (factor ** 2) ** 2)

    def shrink_kernel(self, factor):
        # self.reset_kernel_scale()

        items = self.current_control_points.items()
        self.current_size = self.current_size / factor
        self.current_control_points = {}

        sx = ((1 / factor) ** (self.order + 1))
        sy = ((1 / factor) ** (self.order + 1))

        for i in items:
            current_coord = i[0]
            current_y = i[1]
            self.current_control_points[(current_coord[0] / factor, current_coord[1] / factor)] = current_y / (sx * sy)

    def reset_kernel_scale(self):
        self.current_control_points = self.original_control_points
        self.current_size = self.original_size


    def get_min(self):
        return self.min_x, self.min_y

    def get_max(self):
        return self.max_x, self.max_y

    def get_original_size(self):
        return self.original_size

    def get_current_size(self):
        return self.current_size

    def get_n_control_points(self):
        return len(self.original_control_points.items())

    def get_control_points(self):
        # get original_control points
        xs = self.current_control_points.keys()
        current_size = self.get_current_size()

        # print(xs)
        x_array = []
        y_array = []
        z_array = []

        for x in xs:
            current_x, current_y = x
            # print(f'{current_x} --- {current_y}')
            # current_x = current_x * (1 / current_size)
            # current_y = current_y * (1 / current_size)
            # print(f'{current_x} --- {current_y}')

            current_z = self.current_control_points[x]
            x_array.append(current_x)
            y_array.append(current_y)
            z_array.append(current_z)

        x_array = np.array(x_array)
        y_array = np.array(y_array)
        z_array = np.array(z_array)

        coordinates_array = np.concatenate([x_array[:, None], y_array[:, None]], -1)

        return coordinates_array, z_array

    def get_original_control_points(self):
        # get original_control points
        xs = self.original_control_points.keys()
        current_size = self.get_current_size()

        # print(xs)
        x_array = []
        y_array = []
        z_array = []

        for x in xs:
            current_x, current_y = x
            # print(f'{current_x} --- {current_y}')
            # current_x = current_x * (1 / current_size)
            # current_y = current_y * (1 / current_size)
            # print(f'{current_x} --- {current_y}')

            current_z = self.original_control_points[x]
            x_array.append(current_x)
            y_array.append(current_y)
            z_array.append(current_z)

        x_array = np.array(x_array)
        y_array = np.array(y_array)
        z_array = np.array(z_array)

        coordinates_array = np.concatenate([x_array[:, None], y_array[:, None]], -1)

        return coordinates_array, z_array

    def __str__(self):
        return f'Original: {str(self.original_control_points)} \ncurrent : {self.current_control_points}'


# ----------------------------------------------------------------------------------------------------------------------


class Kernel3d(ABC):
    def __init__(self, input_coordinates=None, zs=None):
        self.original_control_points = {}
        self.current_control_points = {}

        self.original_size = 0
        self.current_size = self.original_size
        self.min_x = 0.0
        self.max_x = 0.0
        self.min_y = 0.0
        self.max_y = 0.0
        self.min_z = 0.0
        self.max_z = 0.0

        if input_coordinates is not None and zs is not None:
            self.original_size = 1
            self.initialize_control_points(input_coordinates, zs)

    def initialize_control_points(self, input_coordinates, zs):

        H, W, Z, D = input_coordinates.shape

        self.min_x = input_coordinates[:, :, :, 0].min()
        self.max_x = input_coordinates[:, :, :, 0].max()

        self.min_y = input_coordinates[:, :, :, 1].min()
        self.max_y = input_coordinates[:, :, :, 1].max()

        self.min_z = input_coordinates[:, :, :, 2].min()
        self.max_z = input_coordinates[:, :, :, 2].max()

        self.original_size = 1
        self.current_size = self.original_size

        for i in range(H):
            for j in range(W):
                for k in range(Z):
                    current_x = input_coordinates[i, j, k]
                    current_y = zs[i, j, k]
                    self.original_control_points[tuple(current_x.tolist())] = current_y

        self.current_control_points = self.original_control_points

    def stretch_kernel(self, factor):
        # self.reset_kernel_scale()

        items = self.current_control_points.items()
        self.current_size = self.current_size * factor
        self.current_control_points = {}

        for i in items:
            current_coord = i[0]
            current_y = i[1]
            self.current_control_points[
                (current_coord[0] * factor,
                 current_coord[1] * factor,
                 current_coord[2] * factor)] = (current_y / (((factor ** 2) ** 2) ** 2))

    def shrink_kernel(self, factor):
        # self.reset_kernel_scale()

        items = self.current_control_points.items()
        self.current_size = self.current_size / factor
        self.current_control_points = {}

        for i in items:
            current_coord = i[0]
            current_y = i[1]
            self.current_control_points[
                (current_coord[0] / factor,
                 current_coord[1] / factor,
                 current_coord[2] / factor)] = current_y * (factor ** 2 ** 3 ** 1)

    def reset_kernel_scale(self):
        self.current_control_points = self.original_control_points
        self.current_size = self.original_size

    def get_min(self):
        return self.min_x, self.min_y, self.min_z

    def get_max(self):
        return self.max_x, self.max_y, self.max_z

    def get_original_size(self):
        return self.original_size

    def get_current_size(self):
        return self.current_size

    def get_n_control_points(self):
        return len(self.original_control_points.items())

    def get_control_points(self):
        xs = self.current_control_points.keys()
        current_size = self.get_current_size()

        # print(xs)
        x_array = []
        y_array = []
        z_array = []
        values = []

        for x in xs:
            current_x, current_y, current_z = x
            current_value = self.current_control_points[x]

            x_array.append(current_x)
            y_array.append(current_y)
            z_array.append(current_z)
            values.append(current_value)

        x_array = np.array(x_array)
        y_array = np.array(y_array)
        z_array = np.array(z_array)
        values = np.array(values)

        coordinates_array = np.concatenate([x_array[:, None], y_array[:, None], z_array[:, None]], -1)

        return coordinates_array, values

    def get_original_control_points(self):
        xs = self.original_control_points.keys()
        current_size = self.get_current_size()

        # print(xs)
        x_array = []
        y_array = []
        z_array = []
        values = []

        for x in xs:
            current_x, current_y, current_z = x
            current_value = self.current_control_points[x]

            x_array.append(current_x)
            y_array.append(current_y)
            z_array.append(current_z)
            values.append(current_value)

        x_array = np.array(x_array)
        y_array = np.array(y_array)
        z_array = np.array(z_array)
        values = np.array(values)

        coordinates_array = np.concatenate([x_array[:, None], y_array[:, None], z_array[:, None]], -1)

        return coordinates_array, values

    def __str__(self):
        return f'Original: {str(self.original_control_points)} \ncurrent : {self.current_control_points}'


class TempKernel3d(ABC):
    def __init__(self, input_coordinates=None, zs=None):
        self.original_control_points = {}
        self.current_control_points = {}

        self.original_size = 0
        self.current_size = self.original_size
        self.min_x = 0.0
        self.max_x = 0.0
        self.min_y = 0.0
        self.max_y = 0.0
        self.min_z = 0.0
        self.max_z = 0.0
        self.order = None

        if input_coordinates is not None and zs is not None:
            self.original_size = 1
            self.initialize_control_points(input_coordinates, zs)

    def initialize_control_points(self, input_coordinates, zs, order):

        H, W, Z, D = input_coordinates.shape

        self.min_x = input_coordinates[:, :, :, 0].min()
        self.max_x = input_coordinates[:, :, :, 0].max()

        self.min_y = input_coordinates[:, :, :, 1].min()
        self.max_y = input_coordinates[:, :, :, 1].max()

        self.min_z = input_coordinates[:, :, :, 2].min()
        self.max_z = input_coordinates[:, :, :, 2].max()
        self.order = order

        self.original_size = 1
        self.current_size = self.original_size

        for i in range(H):
            for j in range(W):
                for k in range(Z):
                    current_x = input_coordinates[i, j, k]
                    current_y = zs[i, j, k]
                    self.original_control_points[tuple(current_x.tolist())] = current_y

        self.current_control_points = self.original_control_points

    def stretch_kernel(self, factor):
        # self.reset_kernel_scale()

        items = self.current_control_points.items()
        self.current_size = self.current_size * factor
        self.current_control_points = {}

        for i in items:
            current_coord = i[0]
            current_y = i[1]
            self.current_control_points[
                (current_coord[0] * factor,
                 current_coord[1] * factor,
                 current_coord[2] * factor)] = (current_y / (((factor ** 2) ** 2) ** 2))

    def shrink_kernel(self, factor):
        # self.reset_kernel_scale()

        items = self.current_control_points.items()
        self.current_size = self.current_size / factor
        self.current_control_points = {}
        sx = ((1 / factor) ** (self.order + 1))
        sy = ((1 / factor) ** (self.order + 1))
        sz = ((1 / factor) ** (self.order + 1))

        for i in items:
            current_coord = i[0]
            current_y = i[1]
            self.current_control_points[
                (current_coord[0] / factor,
                 current_coord[1] / factor,
                 current_coord[2] / factor)] = current_y / (
                        sx * sy * sz)  # current_y * (factor ** 2 ** 3 ** 1) # used to be 2, 2, 2

    def reset_kernel_scale(self):
        self.current_control_points = self.original_control_points
        self.current_size = self.original_size

    def get_min(self):
        return self.min_x, self.min_y, self.min_z

    def get_max(self):
        return self.max_x, self.max_y, self.max_z

    def get_original_size(self):
        return self.original_size

    def get_current_size(self):
        return self.current_size

    def get_n_control_points(self):
        return len(self.original_control_points.items())

    def get_control_points(self):
        xs = self.current_control_points.keys()
        current_size = self.get_current_size()

        # print(xs)
        x_array = []
        y_array = []
        z_array = []
        values = []

        for x in xs:
            current_x, current_y, current_z = x
            current_value = self.current_control_points[x]

            x_array.append(current_x)
            y_array.append(current_y)
            z_array.append(current_z)
            values.append(current_value)

        x_array = np.array(x_array)
        y_array = np.array(y_array)
        z_array = np.array(z_array)
        values = np.array(values)

        coordinates_array = np.concatenate([x_array[:, None], y_array[:, None], z_array[:, None]], -1)

        return coordinates_array, values

    def get_original_control_points(self):
        xs = self.original_control_points.keys()
        current_size = self.get_current_size()

        # print(xs)
        x_array = []
        y_array = []
        z_array = []
        values = []

        for x in xs:
            current_x, current_y, current_z = x
            current_value = self.current_control_points[x]

            x_array.append(current_x)
            y_array.append(current_y)
            z_array.append(current_z)
            values.append(current_value)

        x_array = np.array(x_array)
        y_array = np.array(y_array)
        z_array = np.array(z_array)
        values = np.array(values)

        coordinates_array = np.concatenate([x_array[:, None], y_array[:, None], z_array[:, None]], -1)

        return coordinates_array, values

    def __str__(self):
        return f'Original: {str(self.original_control_points)} \ncurrent : {self.current_control_points}'


# ----------------------------------------------------------------------------------------------------------------------