import matplotlib.pyplot
import torch
import torch.nn as nn
import matplotlib
import math
import torchvision.transforms as tf
import functools
import numpy as np
import matplotlib.pyplot as plt
import deepkan as dk

PC_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('', ['#000000', '#3F003F', '#7E007E',
                                                                   '#4300BD', '#0300FD', '#003F82',
                                                                   '#007D05', '#7CBE00', '#FBFE00',
                                                                   '#FF7F00', '#FF0500'])

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class SplineLinearLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_knots=5, spline_order=3,
                 noise_scale=0.1, base_scale=1.0, spline_scale=1.0,
                 activation=torch.nn.SiLU, grid_epsilon=0.02, grid_range=[-1, 1],
                 standalone_spline_scaling=True):
        super(SplineLinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_knots = num_knots
        self.spline_order = spline_order
        self.grid_epsilon = grid_epsilon
        self.grid_range = grid_range
        self.standalone_spline_scaling = standalone_spline_scaling

        self.register_buffer('knots', self._calculate_knots(grid_range, num_knots, spline_order))
        self.base_weights = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.spline_weights = torch.nn.Parameter(torch.Tensor(output_dim, input_dim, num_knots + spline_order))
        if standalone_spline_scaling:
            self.spline_scales = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))

        self.noise_scale = noise_scale
        self.base_scale = base_scale
        self.spline_scale = spline_scale
        self.activation = activation()

        self._initialize_parameters()

    def _initialize_parameters(self):
        torch.nn.init.xavier_uniform_(self.base_weights, gain=torch.sqrt(torch.tensor(2.0)))
        noise = torch.rand(self.num_knots + 1, self.input_dim, self.output_dim) - 0.5
        self.spline_weights.data.copy_(self.spline_scale * self._initialize_spline_weights(noise))
        if self.standalone_spline_scaling:
            torch.nn.init.xavier_uniform_(self.spline_scales, gain=torch.sqrt(torch.tensor(2.0)))

    def _calculate_knots(self, grid_range, num_knots, spline_order):
        h = (grid_range[1] - grid_range[0]) / num_knots
        knots = torch.arange(-spline_order, num_knots + spline_order + 1) * h + grid_range[0]
        return knots.expand(self.input_dim, -1).contiguous()

    def _initialize_spline_weights(self, noise):
        return self._fit_curve_to_coefficients(self.knots.T[self.spline_order : -self.spline_order], noise)

    def _compute_b_splines(self, x):
        x = x.unsqueeze(-1)
        bases = ((x >= self.knots[:, :-1]) & (x < self.knots[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - self.knots[:, : -(k + 1)]) / (self.knots[:, k:-1] - self.knots[:, : -(k + 1)]) * bases[:, :, :-1] +
                     (self.knots[:, k + 1 :] - x) / (self.knots[:, k + 1 :] - self.knots[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def _fit_curve_to_coefficients(self, x, y):
        A = self._compute_b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def _scaled_spline_weights(self):
        return self.spline_weights * (self.spline_scales.unsqueeze(-1) if self.standalone_spline_scaling else 1.0)

    def forward(self, x):
        base_output = F.linear(self.activation(x), self.base_weights)
        spline_output = F.linear(self._compute_b_splines(x).view(x.size(0), -1),
                                 self._scaled_spline_weights.view(self.output_dim, -1))
        return base_output + spline_output

    @torch.no_grad()
    def _update_knots(self, x, margin=0.01):
        batch = x.size(0)
        splines = self._compute_b_splines(x).permute(1, 0, 2)
        orig_coeff = self._scaled_spline_weights.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        adaptive_knots = x_sorted[torch.linspace(0, batch - 1, self.num_knots + 1, dtype=torch.int64, device=x.device)]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.num_knots
        uniform_knots = torch.arange(self.num_knots + 1, dtype=torch.float32, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin

        knots = self.grid_epsilon * uniform_knots + (1 - self.grid_epsilon) * adaptive_knots
        knots = torch.cat([
            knots[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
            knots,
            knots[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
        ], dim=0)

        self.knots.copy_(knots.T)
        self.spline_weights.data.copy_(self._fit_curve_to_coefficients(x, unreduced_spline_output))

class DeepKAN(torch.nn.Module):
    """
    Initializes the DeepKAN.

    Args:
        input_dim (int): Dimensionality of input data.
        hidden_layers (list): List of hidden layer dimensions (The last one should the target layer)
        num_knots (int): Number of knots for the spline.
        spline_order (int): Order of the spline.
        noise_scale (float): Scale of the noise.
        base_scale (float): Scale of the base weights.
        spline_scale (float): Scale of the spline weights.
        activation (torch.nn.Module): Activation function to use.
        grid_epsilon (float): Epsilon value for the grid.
        grid_range (list): Range of the grid.
    """
    def __init__(self, input_dim, hidden_layers, num_knots=5, spline_order=3,
                 noise_scale=0.1, base_scale=1.0, spline_scale=1.0,
                 activation=torch.nn.SiLU, grid_epsilon=0.02, grid_range=[-1, 1]):
        super(DeepKAN, self).__init__()
        layers = [input_dim] + hidden_layers
        self.layers = torch.nn.ModuleList()
        for in_dim, out_dim in zip(layers, layers[1:]):
            self.layers.append(SplineLinearLayer(in_dim, out_dim, num_knots, spline_order,
                                                 noise_scale, base_scale, spline_scale,
                                                 activation, grid_epsilon, grid_range))

    def forward(self, x, update_knots=False):
        """
        Forward pass of the DeepKAN.

        Args:
            x (torch.Tensor): Input tensor.
            update_knots (bool): Whether to update knots during forward pass.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers:
            if update_knots:
                layer._update_knots(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Computes the regularization loss of the DeepKAN.

        Args:
            regularize_activation (float): Regularization strength for activation.
            regularize_entropy (float): Regularization strength for entropy.

        Returns:
            torch.Tensor: Regularization loss.
        """
        return sum(layer._regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)

class Model(nn.Module):
    def __init__(self, num_classes, num_channels, mean, std):
        super(Model, self).__init__()

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
        def conv_1(): 
            return [nn.Conv2d(num_channels, 64, 1),
                    nn.Conv2d(64, 64, 3),
                    nn.MaxPool2d(2, 2)]
        
        def conv_2(): 
            return [nn.Conv2d(64, 128, 1),
                    nn.Conv2d(128, 128, 3),
                    nn.MaxPool2d(2, 2)]
        
        def conv_3(): 
            return [nn.Conv2d(128, 256, 1),
                    nn.Conv2d(256, 256, 3, 2),
                    nn.Conv2d(256, 256, 1),
                    nn.MaxPool2d(2, 2)]
        
        self.fft_1 = nn.Sequential(*conv_1())
        self.fft_2 = nn.Sequential(*conv_2())
        self.fft_3 = nn.Sequential(*conv_3())
        
        self.pc_1 = nn.Sequential(*conv_1())
        self.pc_2 = nn.Sequential(*conv_2())
        self.pc_3 = nn.Sequential(*conv_3())
        
        self.orig_1 = nn.Sequential(*conv_1())
        self.orig_2 = nn.Sequential(*conv_2())
        self.orig_3 = nn.Sequential(*conv_3())
        
        #Code to find linear layer input size
        #Then use splinelinear layers and deepkan to make kan network
        lin_in = torch.numel(self.fft_3(self.fft_2(self.fft_1(mean[0]))))
        
        linear_fft = [nn.Flatten(),
                      SplineLinearLayer(lin_in, 16),
                      nn.Dropout(0.4)]
        
        linear_pc_orig = [nn.Flatten(),
                          SplineLinearLayer(lin_in, 32),
                          nn.Dropout(0.4)]
        
        # linear = [nn.Linear(3072, 4096),
        #           nn.Linear(4096, num_classes)]

        linear = [DeepKAN(48, [64, num_classes])]
        
        self.linear_fft = nn.Sequential(*linear_fft)
        self.linear_pc_orig = nn.Sequential(*linear_pc_orig)
        self.linear = nn.Sequential(*linear)
        
    def forward(self, x):
        x = (x - self.mean) / self.std
        x = x.permute((1, 0, 2, 3, 4))
        
        fft = self.fft_1(x[2])
        pc = self.pc_1(x[1])
        orig = self.orig_1(x[0])
        
        fft = torch.add(torch.add(pc, orig), fft)
        
        fft = self.fft_2(fft)
        pc = self.pc_2(pc)
        orig = self.orig_2(orig)
        
        fft = torch.add(torch.add(pc, orig), fft)
        
        fft = self.fft_3(fft)
        pc = self.pc_3(pc)
        orig = self.orig_3(orig)
        
        pc_orig = torch.add(pc, orig)
        
        fft = self.linear_fft(fft)
        pc_orig = self.linear_pc_orig(pc_orig)
        
        cat = torch.cat((fft, pc_orig), 1)
        
        return self.linear(cat)

@functools.cache
def highpass_mask(mask_radius, dim):
    mask = torch.ones(dim, dtype=torch.float32)
    mask_radius = np.multiply(dim, mask_radius)
    center = ((dim[0]-1)/2, (dim[1]-1)/2)
    center_tl = np.subtract(np.floor(center), mask_radius).astype(int)
    center_br = np.add(np.ceil(center), mask_radius).astype(int)
    
    for h in range(center_tl[0], center_br[0]):
        for w in range(center_tl[1], center_br[1]):
            h_dist = abs(h-center[0]) / mask_radius[0]
            w_dist = abs(w-center[1]) / mask_radius[1]
            distance = math.sqrt(h_dist**2 + w_dist**2)
            distance = min(1.0, distance)
            # if distance < 1.0:
            #     mask[h][w] = 0.0

            mask[h][w] = distance**6
    return mask

@functools.cache
def bandpass_mask(mask_radii, dim):
    mask = highpass_mask(mask_radii[0], dim) + lowpass_mask(mask_radii[1], dim)
    mask = (mask - 1.0) * -1.0

    return mask

@functools.cache
def lowpass_mask(mask_radius, dim):
    mask = (highpass_mask(mask_radius, dim) - 1.0) * -1.0

    return mask

def pass_filter(img, mask):
    orig_dim = (img.size(0), img.size(1))
    fft = torch.fft.fft2(img)
    fft = torch.fft.fftshift(fft)

    fft = fft*mask
    
    fft = torch.fft.ifftshift(fft)    
    fft = torch.fft.ifft2(fft, orig_dim)
    fft = fft.type(torch.float32)
    
    fft = torch.clamp(fft, 0.0, 1.0)
    
    return fft

def get_tf_function():
    def transform(img):
        img = img.repeat(3, 1, 1, 1)

        img[1] = torch.from_numpy(PC_CMAP(img[1][2])).permute((2,0,1))[:3,:,:]
        
        pass_points = (0.25, 0.125)

        # img[2][0] = pass_filter(img[2][2], lowpass_mask(pass_points[1], img[2][2].shape))
        # img[2][0] = (img[2][0] - torch.mean(img[2][0])) / torch.std(img[2][0])

        # img[2][1] = pass_filter(img[2][2], bandpass_mask(pass_points, img[2][2].shape))
        # img[2][1] = (img[2][1] - torch.mean(img[2][1])) / torch.std(img[2][1])
        
        img[2][2] = pass_filter(img[2][2], highpass_mask(pass_points[0], img[2][2].shape))
        img[2][2] = (img[2][2] - torch.mean(img[2][2])) / torch.std(img[2][2])

        # img[2][0] = torch.zeros(img[2][2].shape)
        # img[2][1] = torch.zeros(img[2][2].shape)
        # img[2][2] = torch.zeros(img[2][2].shape)

        # img[2][0] = lowpass_mask(pass_points[1], img[2][2].shape)
        # img[2][1] = bandpass_mask(pass_points, img[2][2].shape)
        # img[2][2] = highpass_mask(pass_points[0], img[2][2].shape)

        img[2] = torch.from_numpy(PC_CMAP(img[2][2])).permute((2,0,1))[:3,:,:]
        
        return img
    
    return transform