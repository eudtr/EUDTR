import torch
class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.
    chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    remove the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Each causal dilation convolution block is mainly composed of two
    causal dilation convolutions and one residual connection.

    in_channels: Number of input channels.
    out_channels: Number of output channels.
    kernel_size: Kernel size of the applied causal dilation convolutions.
    padding: Zero-padding applied to the left of the input of the
           non-residual convolutions.
    final: If True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    It is mainly the network of causal dilation part, which is mainly used to determine the depth of convolution block of causal dilation

    in_channels: Number of input channels.
    channels: Number of channels processed in the network and of output channels.
    depth: Depth of the network.
    out_channels: Number of output channels.
    kernel_size: Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """
    the structure of encoder, using a causal CNN and the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    in_channels: Number of input channels.
    channels: Number of channels manipulated in the causal CNN.
    depth: Depth of the causal CNN.
    reduced_size: Fixed length to which the output time series of the
           causal CNN is reduced.
    out_channels: Number of output channels.
    kernel_size: Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        )

    def forward(self, x):
        return self.network(x)


class unSqueezeChannels(torch.nn.Module):
    """
    unSqueezes, in a two-dimensional tensor, increase the third dimension.
    """
    def __init__(self):
        super(unSqueezeChannels, self).__init__()

    def forward(self, x):
        return torch.unsqueeze(x, 2)


class Discriminator(torch.nn.Module):

    """
        the structure of discriminator, it is composed of a linear layer 、a Upsample layer and a ConvTranspose1d.

        out_channels: Number of encoder output channels.
        reduced_size: Fixed length to which the output time series of the causal CNN is reduced.
        input_dim: Number of input channels.
        timestep: the length of time series
        kernel_size: Kernel size of the applied ConvTranspose1d.
        stride: stride size of the applied ConvTranspose1d.
        padding: padding size of the applied ConvTranspose1d.
        out_padding: out_padding size of the applied ConvTranspose1d.
    """

    def __init__(self, out_channels, reduced_size, input_dim, timestep, kernel_size=3, stride=1, padding=1, out_padding=0):
        super(Discriminator, self).__init__()
        linear = torch.nn.Linear(out_channels, reduced_size)
        unsqueeze = unSqueezeChannels()
        unsampling = torch.nn.Upsample(size=timestep)
        convT = torch.nn.ConvTranspose1d(reduced_size, input_dim, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=out_padding)
        self.network = torch.nn.Sequential(linear, unsqueeze, unsampling, convT)

    def forward(self, x):
        return self.network(x)
