import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DIP_NN(nn.Module):
    """
    Autencoder for reconstructing object wave and amplitude of LTEM image(s).

    Args:
        num_images: int
            number of input channels, equal to # of images in tfs
        nb_filters: int
            number of filters in 1st convolutional block
            (gets multibplied by 2 in each next block)
        use_dropout: bool
            use / not use dropout in the 3 inner layers
        batch_norm: bool
            use / not use batch normalization after each convolutional layer
        upsampling mode: str
            "bilinear" or "nearest" upsampling method.
            Bilinear is usually more accurate, but adds additional (small)
            randomness; for full reproducibility, consider using 'nearest'
            (this assumes that all other sources of randomness are fixed)
    """

    def __init__(
        self,
        num_images=1,
        nb_filters=16,
        use_dropout=False,
        batch_norm=False,
        upsampling_mode="nearest",
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
        self.cu3p = conv2dblock(
            2, nb_filters * 2, nb_filters * 2, use_batchnorm=batch_norm
        )

        self.upsample_block2p = upsample_block(
            nb_filters * 2, nb_filters, mode=upsampling_mode
        )

        self.cu2p = conv2dblock(1, nb_filters, nb_filters, use_batchnorm=batch_norm)

        self.upsample_block1p = upsample_block(nb_filters, 1, mode=upsampling_mode)

        self.maxpool = F.max_pool2d
        self.concat = torch.cat

    def forward(self, x):
        """Defines a forward path"""
        # with torch.cuda.amp.autocast(): not needed with lightning?
        # Contracting path
        c1 = self.cd1(x)
        d1 = self.maxpool(c1, kernel_size=2, stride=2)
        c2 = self.cd2(d1)
        d2 = self.maxpool(c2, kernel_size=2, stride=2)
        c3 = self.cd3(d2)
        d3 = self.maxpool(c3, kernel_size=2, stride=2)
        c4 = self.cd4(d3)
        d4 = self.maxpool(c4, kernel_size=2, stride=2)
        # Bottleneck layer
        bn = self.bn(d4)

        # Expanding path
        # phase
        # no cu1p because don't want RELU at end and already have the convolution in the end of the upsample block
        # could try replacing the last upsample with a px block like in U-NET idk
        u4_p = self.cu4p(self.upsample_block4p(bn))
        u3_p = self.cu3p(self.upsample_block3p(u4_p))
        u2_p = self.cu2p(self.upsample_block2p(u3_p))
        ph = self.upsample_block1p(u2_p)

        return ph



class conv2dblock(nn.Module):
    """
    Creates block(s) consisting of convolutional
    layer, leaky relu and (optionally) dropout and
    batch normalization
    """

    def __init__(
        self,
        nb_layers,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        use_batchnorm=False,
        lrelu_a=0.01,
        dropout_=0,
        last_sigmoid=False,
        last_tanh=False,
        last_skipReLU=False,
    ):
        """Initializes module parameters"""
        # This is slightly different than ptychoNN in use of filters_in vs filters_out
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
            if last_sigmoid and idx == nb_layers - 1:  # non-RELU activation top layer
                block.append(nn.Sigmoid())
            elif last_tanh and idx == nb_layers - 1:
                block.append(nn.Tanh())
            elif last_skipReLU and idx == nb_layers - 1:
                pass
            else:
                block.append(nn.LeakyReLU(negative_slope=lrelu_a))
                if (
                    use_batchnorm
                ):  # should this be outside for top layer? don't think so
                    block.append(nn.BatchNorm2d(output_channels))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        """Forward path"""
        output = self.block(x)
        return output


class upsample_block(nn.Module):
    """
    Defines upsampling block. The upsampling is performed
    using bilinear or nearest interpolation followed by 1-by-1
    convolution (the latter can be used to reduce
    a number of feature channels).
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        scale_factor=2,
        mode="bilinear",
    ):
        """Initializes module parameters"""
        super(upsample_block, self).__init__()
        assert (
            mode == "bilinear" or mode == "nearest"
        ), "use 'bilinear' or 'nearest' for upsampling mode"
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

    def forward(self, x):
        """Defines a forward path"""
        if self.scale_factor == 2:
            x = self.upsample2x(x)
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


def rng_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_reset(m:nn.Module):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters() # this doesnt require
