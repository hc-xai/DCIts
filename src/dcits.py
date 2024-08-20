import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """
    The Backbone network for the DCITS architecture, consisting of several convolutional layers followed by fully connected layers.

    This backbone network can be configured to use different activation functions, and it adapts the convolutional layers
    based on the input window length (L). The network processes the input through a series of convolutional layers, followed
    by fully connected layers, and applies the specified activation function to the final output.

    Args:
        N (int): The number of time series.
        L (int): The window length.
        activation (str): The type of activation to apply ('softmax' or 'linear').
        in_ch (int, optional): Number of input channels for the convolutional layers. Default is 16.
        temp (float, optional): Temperature parameter for the softmax operation. Default is 1.
    
    Methods:
        forward(x): Passes the input data through the backbone network.

    Returns:
        torch.Tensor: The processed output after applying convolutional and fully connected layers.
    """
    def __init__(self,N,L,activation,in_ch=16,temp=1):
        super().__init__()
        
        self.N=N
        self.L=L
        if temp==None:
            temp=1
        self.temp=temp
        
        
        if activation=='softmax':
            data_ch=1
            self.activation = nn.Sigmoid() 


        elif activation=='linear':            
            data_ch=N
            self.activation = FlattenLinearReshape(N, L)
     
    
        # Define convolutional layers
        kernel_sizes = [(1, L), (N, 1)]
        if L > 1:
            kernel_sizes.append((N, L))
        if L >= 3:
            kernel_sizes.append((N, 3))
            kernel_sizes.append((1, 3))
        if L >= 5:
            kernel_sizes.append((N, 5))
            kernel_sizes.append((1, 5))
        
        self.convs = nn.ModuleList([nn.Conv2d(data_ch, in_ch, ks) for ks in kernel_sizes])

        
        # Adjust fully connected input dimension based on the included layers
        fc_input_dim = 0
        for ks in kernel_sizes:
            if ks[0] == 1:
                    fc_input_dim += (L - ks[1] + 1)*N
            else:
                    fc_input_dim += (L - ks[1] + 1)
                    
        fc_input_dim=fc_input_dim*in_ch
        
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
        
        a=torch.reshape(a,(a.shape[0],self.N,self.N,self.L))
        
        a=self.activation(a/self.temp)
                
        return a
    
    
class DCITS(nn.Module):
    def __init__(self,no_of_timeseries,window_length,in_ch=16,temp=1):
        """
        The DCITS (Deep Convolutional Interpreter of Time Series) model combines two backbones: a focuser and a modeler.

        The focuser uses a softmax-based focusing mechanism to highlight important parts of the input time series.
        The modeler uses a linear transformation to model the relationships in the focused data. The model combines
        these transformations to produce the final output.

        Args:
            no_of_timeseries (int): The number of time series in the input data.
            window_length (int): The length of the time series window.
            in_ch (int, optional): Number of input channels for the convolutional layers. Default is 16.
            temp (float, optional): Temperature parameter for the softmax operation. Default is 1.
    
        Methods:
            forward(x): Passes the input data through the DCITS model to produce the output and intermediate coefficients.
    
        Returns:
            torch.Tensor: The processed output.
            torch.Tensor: The focus weights (f).
            torch.Tensor: The coefficients from the modeler (c).        
        """
        super().__init__()
        
        self.N = no_of_timeseries
        self.L = window_length
        
        self.focuser = Backbone(self.N, self.L, 'softmax', in_ch=in_ch, temp=temp)
        self.modeler = Backbone(self.N, self.L, 'linear', in_ch=in_ch, temp=None)
        
    def forward(self, x):
        """
        Passes the input data through the DCITS model.

        The input is first processed by the focuser to compute the focus weights (f).
        The focused input is then passed through the modeler to compute the coefficients (c).
        The final output is computed by combining the original input, the focus weights, and the coefficients.
        """
        f = self.focuser(x)
        c = self.modeler(f * x)
        x = x * f * c
        x = x.sum(dim=(2, 3))
        
        return x,f,c

    
class DCITSOrder2(nn.Module):
    """
    A neural network module that applies a quadratic (second-order) transformation within the DCITS architecture.

    This module uses a softmax-based focusing mechanism combined with both linear and quadratic coefficient
    transformations to modulate the input time series data.

    Args:
        N (int): The number of time series.
        L (int): The window length.
        in_ch (int, optional): Number of input channels for the convolutional layers. Default is 16.
        temp (float, optional): Temperature parameter for the softmax operation. Default is 1.

    Methods:
        forward(x): Applies the quadratic transformation on the input tensor x.

    Returns:
        torch.Tensor: The processed output.
        torch.Tensor: The focus weights (f).
        torch.Tensor: The linear coefficients (c1).
        torch.Tensor: The quadratic coefficients (c2).
    """
    def __init__(self,no_of_timeseries,window_length,in_ch=16,temp=1):
        super().__init__()
        
        self.N = no_of_timeseries
        self.L = window_length
        
        self.focuser = Backbone(self.N, self.L, 'softmax', in_ch=in_ch, temp=temp)
        self.linear_modeler=Backbone(self.N, self.L, 'linear', in_ch=in_ch, temp=None)
        self.quadratic_modeler=Backbone(self.N, self.L, 'linear', in_ch=in_ch, temp=None)

        
    def forward(self, x):
        """
        Applies the quadratic transformation on the input tensor x.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_channels, N, L).

        Returns:
            torch.Tensor: The processed output of shape (batch_size,).
            torch.Tensor: The focus weights (f).
            torch.Tensor: The linear coefficients (c1).
            torch.Tensor: The quadratic coefficients (c2).
        """
        f=self.focuser(x)
        c1=self.linear_modeler(f*x)
        c2=self.quadratic_modeler(f*x*x)
        x=x*f*c1 + x*x*f*c2
        x=x.sum(dim=(2,3))
        
        return x,f,(c1,c2)
    
class DCITSOrder3(nn.Module):
    """
    A neural network module that applies a cubic (third-order) transformation within the DCITS architecture.

    This module uses a softmax-based focusing mechanism combined with linear, quadratic, and cubic coefficient
    transformations to modulate the input time series data.

    Args:
        N (int): The number of time series.
        L (int): The window length.
        in_ch (int, optional): Number of input channels for the convolutional layers. Default is 16.
        temp (float, optional): Temperature parameter for the softmax operation. Default is 1.

    Methods:
        forward(x): Passes the input data through the third-order transformation process.

    Returns:
        torch.Tensor: The processed output.
        torch.Tensor: The focus weights (f).
        torch.Tensor: The linear coefficients (c1).
        torch.Tensor: The quadratic coefficients (c2).
        torch.Tensor: The cubic coefficients (c3).
    """
    def __init__(self,no_of_timeseries,window_length,in_ch=16,temp=1):
        super().__init__()
        
        self.N = no_of_timeseries
        self.L = window_length
        
        self.focuser = Backbone(self.N, self.L, 'softmax', in_ch=in_ch, temp=temp)
        self.linear_modeler=Backbone(self.N, self.L, 'linear', in_ch=in_ch, temp=None)
        self.quadratic_modeler=Backbone(self.N, self.L, 'linear', in_ch=in_ch, temp=None)
        self.cubic_modeler=Backbone(self.N, self.L, 'linear', in_ch=in_ch, temp=None)

        
    def forward(self, x):
        f=self.focuser(x)
        c1=self.linear_modeler(f*x)
        c2=self.quadratic_modeler(f*x*x)
        c3=self.cubic_modeler(f*x*x*x)
        x=x*f*c1 + x*x*f*c2 + x*x*x*f*c3
        x=x.sum(dim=(2,3))
        
        return x,f,(c1,c2,c3)