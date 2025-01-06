import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class FlattenLinearReshape(nn.Module):
    def __init__(self, N, L):
        """
        Custom module to flatten, apply linear transformation and reshape.
        """
        super().__init__()
        self.N = N
        self.L = L
        self.lin = nn.Linear(N * N * L, N * N * L)
        
    def forward(self, x):
        x = x.flatten(1)
        x = self.lin(x)
        x = x.view(x.shape[0], self.N, self.N, self.L)
        return x
    
class Backbone(nn.Module):
    def __init__(self, N, L, activation, in_ch=16, temperature=1):
        super().__init__()
        
        self.N = N
        self.L = L
        self.temperature = temperature or 1
        
        # Set data channels based on activation type
        if activation == 'sigmoid':
            data_ch = 1
            self.activation = FlattenLinearReshape(N, L)  # Changed: Use same as modeler
            self.final_activation = nn.Sigmoid()  # Add sigmoid after
        elif activation == 'linear':
            data_ch = N
            self.activation = FlattenLinearReshape(N, L)
            self.final_activation = None
        
        # Define convolutional layers
        kernel_sizes = [(1, L), (N, 1)]
        if L > 1:
            kernel_sizes.append((N, L))
        if L >= 3:
            kernel_sizes.extend([(N, 3), (1, 3)])
        if L >= 5:
            kernel_sizes.extend([(N, 5), (1, 5)])
        
        self.convs = nn.ModuleList([nn.Conv2d(data_ch, in_ch, ks) for ks in kernel_sizes])
        
        # Adjust fully connected input dimension
        fc_input_dim = sum(
            in_ch * ((L - ks[1] + 1) * N if ks[0] == 1 else (L - ks[1] + 1))
            for ks in kernel_sizes
        )
        
        # Keep same fc_layers for both
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, N * N * L),
            nn.Tanh()
        )

    def forward(self, x):
        conv_outs = [conv(x).flatten(1) for conv in self.convs]
        a = torch.cat(conv_outs, dim=1)
        a = self.fc_layers(a)
        a = a.view(-1, self.N, self.N, self.L)
        a = self.activation(a / self.temperature)
        if self.final_activation:
            a = self.final_activation(a)
        return a

class DCITS(nn.Module):
    def __init__(self, no_of_timeseries, window_length, order=None, temperature=1, in_ch=16):
        """
        DCITS model generalized to handle arbitrary order terms specified by 'order' list.

        Args:
            no_of_timeseries (int): Number of time series (N).
            window_length (int): Window length (L).
            order (list, optional): List indicating which orders to include.
                                    Example: [1,1,0,1] includes bias (order 0), linear (order 1),
                                    excludes quadratic (order 2), and includes cubic (order 3).
            temperature (float, optional): Temperature parameter for the sigmoid operation.
            in_ch (int, optional): Number of input channels for the convolutional layers.
        """
        super().__init__()
        self.N = no_of_timeseries
        self.L = window_length

        if order is None:
            self.order = [1, 1]
        else:
            self.order = order

        self.bias = self.order[0] == 1

        self.modelers = nn.ModuleDict()
        self.focuseres = nn.ModuleDict()
        for i, include_term in enumerate(self.order):
            if include_term:
                if i == 0:
                    self.bias_focuser = Backbone(self.N, 1, 'sigmoid', in_ch=in_ch, temperature=temperature)
                    self.bias_modeler = Backbone(self.N, 1, 'linear', in_ch=in_ch, temperature=None)
                    diagonal_mask = torch.eye(no_of_timeseries)
                    self.register_buffer('diagonal_mask', diagonal_mask.unsqueeze(-1))
                else:
                    self.focuseres[f'a{i}'] = Backbone(self.N, self.L, 'sigmoid', in_ch=in_ch, temperature=temperature)
                    self.modelers[f'c{i}'] = Backbone(self.N, self.L, 'linear', in_ch=in_ch, temperature=None)

    def forward(self, x, order=None):
        batch_size = x.size(0)
        if order is None:
            inference_order = self.order
        else:
            inference_order = order
            for idx, include_term in enumerate(inference_order):
                if include_term:
                    if idx >= len(self.order) or not self.order[idx]:
                        warnings.warn(f"Order {idx} was not included during training and will be ignored.")
                        inference_order[idx] = 0

        x_total = 0
        f_list = []
        c_list = []

        if self.bias and inference_order[0]:
            ones = torch.ones((batch_size, 1, self.N, 1), device=x.device)
            f_bias = self.bias_focuser(ones)
            c_bias = self.bias_modeler(f_bias)
            f_bias = f_bias * self.diagonal_mask
            c_bias = c_bias * self.diagonal_mask
            x_bias = ones * f_bias * c_bias
            x_bias_sum = x_bias.sum(dim=(2, 3))
            x_total += x_bias_sum
            f_list.append(f_bias)
            c_list.append(c_bias)
        else:
            f_list.append(None)
            c_list.append(None)

        x_expanded = x.repeat(1, self.N, 1, 1)
        x_power_expanded = x_expanded
        x_power = x

        for i in range(1, len(self.order)):
            if i < len(inference_order) and inference_order[i]:
                if self.order[i]:
                    f_i = self.focuseres[f'a{i}'](x_power)
                    c_i = self.modelers[f'c{i}'](f_i * x_power_expanded)
                    x_term = x_power_expanded * f_i * c_i
                    x_term_sum = x_term.sum(dim=(2, 3))
                    x_total += x_term_sum
                    f_list.append(f_i)
                    c_list.append(c_i)
                else:
                    f_list.append(None)
                    c_list.append(None)
            else:
                f_list.append(None)
                c_list.append(None)
            x_power_expanded = x_power_expanded * x_expanded
            x_power = x_power * x

        return x_total, f_list, c_list