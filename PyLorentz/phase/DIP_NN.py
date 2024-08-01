import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DIP_NN(nn.Module):
    """
    Autoencoder for reconstructing object wave and amplitude of LTEM images.

    Args:
        num_images: int, number of input channels, equal to number of images in TFS.
        nb_filters: int, number of filters in the first convolutional block.
        use_dropout: bool, whether to use dropout in the inner layers.
        batch_norm: bool, whether to use batch normalization after each convolutional layer.
        upsampling_mode: str, "bilinear" or "nearest" upsampling method.
    """

    def __init__(
        self,
        num_images: int = 1,
        nb_filters: int = 16,
        use_dropout: bool = False,
        batch_norm: bool = False,
        upsampling_mode: str = "nearest",
    ):
        super().__init__()
        self.num_images = num_images
        self.nb_filters = nb_filters
        self.use_dropout = use_dropout
        self.batch_norm = batch_norm
        self.upsampling_mode = upsampling_mode
        dropout_vals = [0.1, 0.2, 0.1] if use_dropout else [0, 0, 0]

        self.cd1 = conv2dblock(
            nb_layers=2,
            input_channels=self.num_images,
            output_channels=nb_filters,
            use_batchnorm=batch_norm,
        )

        self.cd2 = conv2dblock(2, nb_filters, nb_filters * 2, use_batchnorm=batch_norm)

        self.cd3 = conv2dblock(
            2,
            nb_filters * 2,
            nb_filters * 4,
            use_batchnorm=batch_norm,
            dropout_=dropout_vals[0],
        )

        self.cd4 = conv2dblock(
            2,
            nb_filters * 4,
            nb_filters * 8,
            use_batchnorm=batch_norm,
            dropout_=dropout_vals[0],
        )

        self.bn = conv2dblock(
            2,
            nb_filters * 8,
            nb_filters * 8,
            use_batchnorm=batch_norm,
            dropout_=dropout_vals[1],
        )

        self.upsample_block4p = upsample_block(
            nb_filters * 8, nb_filters * 4, mode=upsampling_mode
        )

        self.cu4p = conv2dblock(
            2,
            nb_filters * 4,
            nb_filters * 4,
            use_batchnorm=batch_norm,
            dropout_=dropout_vals[2],
        )

        self.upsample_block3p = upsample_block(
            nb_filters * 4, nb_filters * 2, mode=upsampling_mode
        )
        self.cu3p = conv2dblock(2, nb_filters * 2, nb_filters * 2, use_batchnorm=batch_norm)

        self.upsample_block2p = upsample_block(nb_filters * 2, nb_filters, mode=upsampling_mode)

        self.cu2p = conv2dblock(1, nb_filters, nb_filters, use_batchnorm=batch_norm)

        self.upsample_block1p = upsample_block(nb_filters, self.num_images, mode=upsampling_mode)

        self.maxpool = F.max_pool2d
        self.concat = torch.cat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DIP network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the network.
        """
        c1 = self.cd1(x)
        d1 = self.maxpool(c1, kernel_size=2, stride=2)
        c2 = self.cd2(d1)
        d2 = self.maxpool(c2, kernel_size=2, stride=2)
        c3 = self.cd3(d2)
        d3 = self.maxpool(c3, kernel_size=2, stride=2)
        c4 = self.cd4(d3)
        d4 = self.maxpool(c4, kernel_size=2, stride=2)
        bn = self.bn(d4)

        u4_p = self.cu4p(self.upsample_block4p(bn))
        u3_p = self.cu3p(self.upsample_block3p(u4_p))
        u2_p = self.cu2p(self.upsample_block2p(u3_p))
        ph = self.upsample_block1p(u2_p)

        return ph


class conv2dblock(nn.Module):
    """
    A block consisting of convolutional layers with optional batch normalization and dropout.

    Args:
        nb_layers: int, number of convolutional layers.
        input_channels: int, number of input channels.
        output_channels: int, number of output channels.
        kernel_size: int, size of the convolutional kernel.
        stride: int, stride of the convolution.
        padding: int, padding for the convolution.
        use_batchnorm: bool, whether to use batch normalization.
        lrelu_a: float, negative slope for the Leaky ReLU activation.
        dropout_: float, dropout rate.
        last_sigmoid: bool, whether to use a sigmoid activation on the last layer.
        last_tanh: bool, whether to use a tanh activation on the last layer.
        last_skipReLU: bool, whether to skip ReLU activation on the last layer.
    """

    def __init__(
        self,
        nb_layers: int,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = False,
        lrelu_a: float = 0.01,
        dropout_: float = 0,
        last_sigmoid: bool = False,
        last_tanh: bool = False,
        last_skipReLU: bool = False,
    ):
        super(conv2dblock, self).__init__()
        block = []
        for idx in range(nb_layers):
            input_channels = output_channels if idx > 0 else input_channels
            block.append(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            if dropout_ > 0:
                block.append(nn.Dropout(dropout_))
            if last_sigmoid and idx == nb_layers - 1:
                block.append(nn.Sigmoid())
            elif last_tanh and idx == nb_layers - 1:
                block.append(nn.Tanh())
            elif last_skipReLU and idx == nb_layers - 1:
                pass
            else:
                block.append(nn.LeakyReLU(negative_slope=lrelu_a))
                if use_batchnorm:
                    block.append(nn.BatchNorm2d(output_channels))
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the conv2dblock.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the block.
        """
        return self.block(x)


class upsample_block(nn.Module):
    """
    Upsampling block using interpolation followed by a convolution.

    Args:
        input_channels: int, number of input channels.
        output_channels: int, number of output channels.
        scale_factor: int, factor by which to scale the input.
        mode: str, interpolation mode, either "bilinear" or "nearest".
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        scale_factor: int = 2,
        mode: str = "bilinear",
    ):
        super(upsample_block, self).__init__()
        assert mode in ["bilinear", "nearest"], "Mode must be 'bilinear' or 'nearest'."
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.upsample2x = nn.ConvTranspose2d(
            input_channels,
            input_channels,
            kernel_size=3,
            stride=2,
            padding=(1, 1),
            output_padding=(1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the upsample_block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after upsampling and convolution.
        """
        if self.scale_factor == 2:
            x = self.upsample2x(x)
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


def rng_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed: The seed value to use.

    Returns:
        None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_reset(m: nn.Module) -> None:
    """
    Reset the weights of a given module.

    Args:
        m: The neural network module to reset.

    Returns:
        None
    """
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()
