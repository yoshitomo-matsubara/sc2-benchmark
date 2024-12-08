import torch
from compressai.entropy_models import GaussianConditional
from compressai.layers import GDN1
from compressai.models import CompressionModel
from compressai.models.google import get_scale_table
from compressai.models.utils import update_registered_buffers
from torch import nn
from torchdistill.common.constant import def_logger

logger = def_logger.getChild(__name__)
LAYER_CLASS_DICT = dict()
LAYER_FUNC_DICT = dict()


def register_layer_class(cls):
    """
    Registers a layer class.

    :param cls: layer class to be registered
    :type cls: class
    :return: registered layer class
    :rtype: class
    """
    LAYER_CLASS_DICT[cls.__name__] = cls
    return cls


def register_layer_func(func):
    """
    Registers a function to build a layer module.

    :param func: function to build a layer module
    :type func: typing.Callable
    :return: registered function
    :rtype: typing.Callable
    """
    LAYER_FUNC_DICT[func.__name__] = func
    return func


class SimpleBottleneck(nn.Module):
    """
    Simple neural encoder-decoder that treats encoder's output as bottleneck.

    The forward path is encoder -> compressor (if provided) -> decompressor (if provided) -> decoder.

    :param encoder: encoder
    :type encoder: nn.Module
    :param decoder: decoder
    :type decoder: nn.Module
    :param encoder: module to compress the encoded data
    :type encoder: nn.Module or None
    :param decoder: module to decompresse the compressed data
    :type decoder: nn.Module or None
    """
    def __init__(self, encoder, decoder, compressor=None, decompressor=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.compressor = compressor
        self.decompressor = decompressor

    def encode(self, x):
        """
        Encode the input data.

        :param x: input batch
        :type x: torch.Tensor
        :return: dict of encoded (and compressed if `compressor` is provided)
        :rtype: dict
        """
        z = self.encoder(x)
        if self.compressor is not None:
            z = self.compressor(z)
        return {'z': z}

    def decode(self, z):
        """
        Decode the encoded data.

        :param z: encoded data
        :type z: torch.Tensor
        :return: decoded data
        :rtype: torch.Tensor
        """
        if self.decompressor is not None:
            z = self.decompressor(z)
        return self.decoder(z)

    def forward(self, x):
        if not self.training:
            encoded_obj = self.encode(x)
            decoded_obj = self.decode(**encoded_obj)
            return decoded_obj

        z = self.encoder(x)
        return self.decoder(z)

    def update(self):
        """
        Shows a message that this module has no updatable parameters for entropy coding.

        Dummy function to be compatible with other layers.
        """
        logger.info('This module has no updatable parameters for entropy coding')


@register_layer_func
def larger_resnet_bottleneck(bottleneck_channel=12, bottleneck_idx=7, output_channel=256,
                             compressor_transform=None, decompressor_transform=None):
    """
    Builds a bottleneck layer ResNet-based encoder and decoder (24 layers in total).

    Compatible with ResNet-50, -101, and -152.

    Yoshitomo Matsubara, Marco Levorato: `"Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks" <https://arxiv.org/abs/2007.15818>`_ @ ICPR 2020 (2021)

    :param bottleneck_channel: number of channels for the bottleneck point
    :type bottleneck_idx: int
    :param bottleneck_idx: number of the first layers to be used as an encoder (the remaining layers are for decoder)
    :type bottleneck_idx: int
    :param output_channel: number of output channels for decoder's output
    :type output_channel: int
    :param compressor_transform: compressor transform
    :type compressor_transform: nn.Module or None
    :param decompressor_transform: decompressor transform
    :type decompressor_transform: nn.Module or None
    :return: bottleneck layer consisting of encoder and decoder
    :rtype: SimpleBottleneck
    """
    modules = [
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(bottleneck_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=1)
    ]
    encoder = nn.Sequential(*modules[:bottleneck_idx])
    decoder = nn.Sequential(*modules[bottleneck_idx:])
    return SimpleBottleneck(encoder, decoder, compressor_transform, decompressor_transform)


@register_layer_func
def larger_densenet_bottleneck(bottleneck_channel=12, bottleneck_idx=7,
                               compressor_transform=None, decompressor_transform=None):
    """
    Builds a bottleneck layer DenseNet-based encoder and decoder (23 layers in total).

    Compatible with DenseNet-169 and -201.

    Yoshitomo Matsubara, Davide Callegaro, Sabur Baidya, Marco Levorato, Sameer Singh: `"Head Network Distillation: Splitting Distilled Deep Neural Networks for Resource-constrained Edge Computing Systems" <https://ieeexplore.ieee.org/document/9265295>`_ @ IEEE Access (2020)

    :param bottleneck_channel: number of channels for the bottleneck point
    :type bottleneck_idx: int
    :param bottleneck_idx: number of the first layers to be used as an encoder (the remaining layers are for decoder)
    :type bottleneck_idx: int
    :param compressor_transform: compressor transform
    :type compressor_transform: nn.Module or None
    :param decompressor_transform: decompressor transform
    :type decompressor_transform: nn.Module or None
    :return: bottleneck layer consisting of encoder and decoder
    :rtype: SimpleBottleneck
    """
    modules = [
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(bottleneck_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 256, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    ]
    encoder = nn.Sequential(*modules[:bottleneck_idx])
    decoder = nn.Sequential(*modules[bottleneck_idx:])
    return SimpleBottleneck(encoder, decoder, compressor_transform, decompressor_transform)


@register_layer_func
def inception_v3_bottleneck(bottleneck_channel=12, bottleneck_idx=7,
                            compressor_transform=None, decompressor_transform=None):
    """
    Builds a bottleneck layer InceptionV3-based encoder and decoder (17 layers in total).

    Yoshitomo Matsubara, Davide Callegaro, Sabur Baidya, Marco Levorato, Sameer Singh: `"Head Network Distillation: Splitting Distilled Deep Neural Networks for Resource-constrained Edge Computing Systems" <https://ieeexplore.ieee.org/document/9265295>`_ @ IEEE Access (2020)

    :param bottleneck_channel: number of channels for the bottleneck point
    :type bottleneck_idx: int
    :param bottleneck_idx: number of the first layers to be used as an encoder (the remaining layers are for decoder)
    :type bottleneck_idx: int
    :param compressor_transform: compressor transform
    :type compressor_transform: nn.Module or None
    :param decompressor_transform: decompressor transform
    :type decompressor_transform: nn.Module or None
    :return: bottleneck layer consisting of encoder and decoder
    :rtype: SimpleBottleneck
    """
    modules = [
        nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(bottleneck_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(bottleneck_channel, 256, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 192, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=1)
    ]
    encoder = nn.Sequential(*modules[:bottleneck_idx])
    decoder = nn.Sequential(*modules[bottleneck_idx:])
    return SimpleBottleneck(encoder, decoder, compressor_transform, decompressor_transform)


class EntropyBottleneckLayer(CompressionModel):
    """
    An entropy bottleneck layer as a simple `CompressionModel` in `compressai`.

    Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, Nick Johnston: `"Variational Image Compression with a Scale Hyperprior" <https://openreview.net/forum?id=rkcQFMZRb>`_ @ ICLR 2018 (2018)

    :param kwargs: kwargs for `CompressionModel` in `compressai`
    :type kwargs: dict
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.updated = False

    def forward(self, x):
        return self.entropy_bottleneck(x)

    def compress(self, x):
        """
        Compresses input data.

        :param x: input data
        :type x: torch.Tensor
        :return: entropy-coded compressed data ('strings' as key) and shape of the input data ('shape' as key)
        :rtype: dict
        """
        strings = self.entropy_bottleneck.compress(x)
        return {'strings': [strings], 'shape': x.size()[-2:]}

    def decompress(self, strings, shape):
        """
        Dempresses compressed data.

        :param strings: entropy-coded compressed data
        :type strings: list[str]
        :param shape: shape of the input data
        :type shape: list[int]
        :return: decompressed data
        :rtype: torch.Tensor
        """
        assert isinstance(strings, list) and len(strings) == 1
        return self.entropy_bottleneck.decompress(strings[0], shape)

    def update(self, force=False):
        """
        Updates compression-specific parameters like `CompressAI models do <https://interdigitalinc.github.io/CompressAI/models.html#compressai.models.CompressionModel.update>`_.

        :param force: if True, overwrites previous values
        :type force: bool
        :return: True if one of the EntropyBottlenecks was updated
        :rtype: bool
        """
        self.updated = True
        return super().update(force=force)


class BaseBottleneck(CompressionModel):
    """
    An abstract class for entropy bottleneck-based layer.

    :param entropy_bottleneck_channels: number of entropy bottleneck channels
    :type entropy_bottleneck_channels: int
    """
    def __init__(self, entropy_bottleneck_channels):
        super().__init__(entropy_bottleneck_channels=entropy_bottleneck_channels)
        self.updated = False

    def encode(self, *args, **kwargs):
        """
        Encodes data.

        This should be overridden by all subclasses.
        """
        raise NotImplementedError()

    def decode(self, *args, **kwargs):
        """
        Decodes encoded data.

        This should be overridden by all subclasses.
        """
        raise NotImplementedError()

    def forward(self, *args):
        raise NotImplementedError()

    def update(self, force=False):
        """
        Updates compression-specific parameters like `CompressAI models do <https://interdigitalinc.github.io/CompressAI/models.html#compressai.models.CompressionModel.update>`_.

        :param force: if True, overwrites previous values
        :type force: bool
        :return: True if one of the EntropyBottlenecks was updated
        :rtype: bool
        """
        self.updated = True
        return super().update(force=force)


@register_layer_class
class FPBasedResNetBottleneck(BaseBottleneck):
    """
    Factorized Prior(FP)-based encoder-decoder designed to create bottleneck for ResNet and variants.

    - Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, Nick Johnston: `"Variational Image Compression with a Scale Hyperprior" <https://openreview.net/forum?id=rkcQFMZRb>`_ @ ICLR 2018 (2018)
    - Yoshitomo Matsubara, Ruihan Yang, Marco Levorato, Stephan Mandt: `"Supervised Compression for Resource-Constrained Edge Computing Systems" <https://openaccess.thecvf.com/content/WACV2022/html/Matsubara_Supervised_Compression_for_Resource-Constrained_Edge_Computing_Systems_WACV_2022_paper.html>`_ @ WACV 2022 (2022)
    - Yoshitomo Matsubara, Ruihan Yang, Marco Levorato, Stephan Mandt: `"SC2 Benchmark: Supervised Compression for Split Computing" <https://openreview.net/forum?id=p28wv4G65d>`_ @ TMLR (2023)

    :param num_input_channels: number of input channels
    :type num_input_channels: int
    :param num_bottleneck_channels: number of bottleneck channels
    :type num_bottleneck_channels: int
    :param num_target_channels: number of output channels for decoder's output
    :type num_target_channels: int
    :param encoder_channel_sizes: list of 4 numbers of channels for encoder
    :type encoder_channel_sizes: list[int] or None
    :param decoder_channel_sizes: list of 4 numbers of channels for decoder
    :type decoder_channel_sizes: list[int] or None
    """
    def __init__(self, num_input_channels=3, num_bottleneck_channels=24, num_target_channels=256,
                 encoder_channel_sizes=None, decoder_channel_sizes=None):
        if encoder_channel_sizes is None:
            encoder_channel_sizes = \
                [num_input_channels, num_bottleneck_channels * 4, num_bottleneck_channels * 2, num_bottleneck_channels]

        if decoder_channel_sizes is None:
            decoder_channel_sizes = \
                [encoder_channel_sizes[-1], num_target_channels * 2, num_target_channels, num_target_channels]

        super().__init__(entropy_bottleneck_channels=num_bottleneck_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(encoder_channel_sizes[0], encoder_channel_sizes[1],
                      kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(encoder_channel_sizes[1]),
            nn.Conv2d(encoder_channel_sizes[1], encoder_channel_sizes[2],
                      kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(encoder_channel_sizes[2]),
            nn.Conv2d(encoder_channel_sizes[2], encoder_channel_sizes[3],
                      kernel_size=2, stride=1, padding=0, bias=False)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_channel_sizes[0], decoder_channel_sizes[1],
                      kernel_size=2, stride=1, padding=1, bias=False),
            GDN1(decoder_channel_sizes[1], inverse=True),
            nn.Conv2d(decoder_channel_sizes[1], decoder_channel_sizes[2],
                      kernel_size=2, stride=1, padding=0, bias=False),
            GDN1(decoder_channel_sizes[2], inverse=True),
            nn.Conv2d(decoder_channel_sizes[2], decoder_channel_sizes[3],
                      kernel_size=2, stride=1, padding=1, bias=False)
        )

    def encode(self, x, **kwargs):
        """
        Encodes input data.

        :param x: input data
        :type x: torch.Tensor
        :return: entropy-coded compressed data ('strings' as key) and shape of the input data ('shape' as key)
        :rtype: dict
        """
        latent = self.encoder(x)
        latent_strings = self.entropy_bottleneck.compress(latent)
        return {'strings': [latent_strings], 'shape': latent.size()[-2:]}

    def decode(self, strings, shape):
        """
        Decodes encoded data.

        :param strings: entropy-coded compressed data
        :type strings: list[str]
        :param shape: shape of the input data
        :type shape: list[int]
        :return: decompressed data
        :rtype: torch.Tensor
        """
        latent_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        return self.decoder(latent_hat)

    def _get_means(self, x):
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        return medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))

    def _forward2train(self, x):
        encoded_obj = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(encoded_obj)
        decoded_obj = self.decoder(y_hat)
        return decoded_obj

    def forward(self, x):
        # if fine-tune or evaluate after "update"
        if self.updated:
            if not self.training:
                encoded_obj = self.encode(x)
                decoded_obj = self.decode(**encoded_obj)
                return decoded_obj

            encoded_output = self.encoder(x)
            decoder_input =\
                self.entropy_bottleneck.dequantize(
                    self.entropy_bottleneck.quantize(encoded_output, 'dequantize', self._get_means(encoded_output))
                )
            decoder_input = decoder_input.detach()
            return self.decoder(decoder_input)
        return self._forward2train(x)


@register_layer_class
class SHPBasedResNetBottleneck(BaseBottleneck):
    """
    Scale Hyperprior(SHP)-based bottleneck for ResNet and variants.

    - Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, Nick Johnston: `"Variational Image Compression with a Scale Hyperprior" <https://openreview.net/forum?id=rkcQFMZRb>`_ @ ICLR 2018 (2018)
    - Yoshitomo Matsubara, Ruihan Yang, Marco Levorato, Stephan Mandt: `"SC2 Benchmark: Supervised Compression for Split Computing" <https://openreview.net/forum?id=p28wv4G65d>`_ @ TMLR (2023)

    :param num_input_channels: number of input channels
    :type num_input_channels: int
    :param num_latent_channels: number of latent channels
    :type num_latent_channels: int
    :param num_bottleneck_channels: number of bottleneck channels
    :type num_bottleneck_channels: int
    :param num_target_channels: number of output channels for decoder's output
    :type num_target_channels: int
    :param h_a: parametric transform :math:`h_a`
    :type h_a: nn.Module or None
    :param h_s: parametric transform :math:`h_s`
    :type h_s: nn.Module or None
    :param g_a_channel_sizes: list of 4 numbers of channels for parametric transform :math:`g_a`
    :type g_a_channel_sizes: list[int] or None
    :param g_s_channel_sizes: list of 4 numbers of channels for parametric transform :math:`g_s`
    :type g_s_channel_sizes: list[int] or None
    """
    def __init__(self, num_input_channels=3, num_latent_channels=16,
                 num_bottleneck_channels=24, num_target_channels=256, h_a=None, h_s=None,
                 g_a_channel_sizes=None, g_s_channel_sizes=None):
        if g_a_channel_sizes is None:
            g_a_channel_sizes = \
                [num_input_channels, num_bottleneck_channels * 4, num_bottleneck_channels * 2, num_bottleneck_channels]
        else:
            num_bottleneck_channels = g_a_channel_sizes[3]

        if g_s_channel_sizes is None:
            g_s_channel_sizes = \
                [g_a_channel_sizes[-1], num_target_channels * 2, num_target_channels, num_target_channels]
        super().__init__(entropy_bottleneck_channels=num_latent_channels)
        self.g_a = nn.Sequential(
            nn.Conv2d(g_a_channel_sizes[0], g_a_channel_sizes[1],
                      kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(g_a_channel_sizes[1]),
            nn.Conv2d(g_a_channel_sizes[1], g_a_channel_sizes[2],
                      kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(g_a_channel_sizes[2]),
            nn.Conv2d(g_a_channel_sizes[2], g_a_channel_sizes[3],
                      kernel_size=2, stride=1, padding=0, bias=False)
        )

        self.g_s = nn.Sequential(
            nn.Conv2d(g_s_channel_sizes[0], g_s_channel_sizes[1], kernel_size=2, stride=1, padding=1, bias=False),
            GDN1(g_s_channel_sizes[1], inverse=True),
            nn.Conv2d(g_s_channel_sizes[1], g_s_channel_sizes[2], kernel_size=2, stride=1, padding=0, bias=False),
            GDN1(g_s_channel_sizes[2], inverse=True),
            nn.Conv2d(g_s_channel_sizes[2], g_s_channel_sizes[3], kernel_size=2, stride=1, padding=1, bias=False)
        )

        self.h_a = nn.Sequential(
            nn.Conv2d(num_bottleneck_channels, num_latent_channels, kernel_size=5, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_latent_channels, num_latent_channels, kernel_size=5, stride=2, padding=2, bias=False)
        ) if h_a is None else h_a

        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(num_latent_channels, num_latent_channels,
                               kernel_size=5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(num_latent_channels, num_latent_channels,
                               kernel_size=5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_latent_channels, num_bottleneck_channels,
                      kernel_size=5, stride=1, padding=0, bias=False)
        ) if h_s is None else h_s

        self.gaussian_conditional = GaussianConditional(None)
        self.num_latent_channels = num_latent_channels
        self.num_bottleneck_channels = num_bottleneck_channels

    def encode(self, x, **kwargs):
        """
        Encodes input data.

        :param x: input data
        :type x: torch.Tensor
        :return: entropy-coded compressed data ('strings' as key) and shape of the input data ('shape' as key)
        :rtype: dict
        """
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_shape = z.size()[-2:]
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)
        scales_hat = self.h_s(z_hat)
        indices = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indices)
        return {'strings': [y_strings, z_strings], 'shape': z_shape}

    def decode(self, strings, shape):
        """
        Decodes encoded data.

        :param strings: entropy-coded compressed data
        :type strings: list[str]
        :param shape: shape of the input data
        :type shape: list[int]
        :return: decompressed data
        :rtype: torch.Tensor
        """
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indices = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indices, z_hat.dtype)
        return self.g_s(y_hat)

    def _get_means(self, x):
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        return medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))

    def _forward2train(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        return self.g_s(y_hat)

    def forward(self, x):
        # if fine-tune or evaluate after "update"
        if self.updated:
            if not self.training:
                encoded_obj = self.encode(x)
                decoded_obj = self.decode(**encoded_obj)
                return decoded_obj

            y = self.g_a(x)
            y_hat = self.gaussian_conditional.dequantize(
                self.gaussian_conditional.quantize(y, 'dequantize', self._get_means(y))
            )
            y_hat = y_hat.detach()
            return self.g_s(y_hat)
        return self._forward2train(x)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()

        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        self.updated = True
        return updated

    def load_state_dict(self, state_dict, **kwargs):
        """
        Updates registered buffers and loads parameters.

        :param state_dict: dict containing parameters and persistent buffers
        :type state_dict: dict
        """
        update_registered_buffers(
            self.gaussian_conditional,
            'gaussian_conditional',
            ['_quantized_cdf', '_offset', '_cdf_length', 'scale_table'],
            state_dict,
        )
        super().load_state_dict(state_dict)


@register_layer_class
class MSHPBasedResNetBottleneck(SHPBasedResNetBottleneck):
    """
    Mean-Scale Hyperprior(MSHP)-based bottleneck for ResNet and variants.

    - David Minnen, Johannes Ballé, George Toderici: `"Joint Autoregressive and Hierarchical Priors for Learned Image Compression" <https://proceedings.neurips.cc/paper/2018/hash/53edebc543333dfbf7c5933af792c9c4-Abstract.html>`_ @ NeurIPS 2018 (2018)
    - Yoshitomo Matsubara, Ruihan Yang, Marco Levorato, Stephan Mandt: `"SC2 Benchmark: Supervised Compression for Split Computing" <https://openreview.net/forum?id=p28wv4G65d>`_ @ TMLR (2023)

    :param num_input_channels: number of input channels
    :type num_input_channels: int
    :param num_latent_channels: number of latent channels
    :type num_latent_channels: int
    :param num_bottleneck_channels: number of bottleneck channels
    :type num_bottleneck_channels: int
    :param num_target_channels: number of output channels for decoder's output
    :type num_target_channels: int
    :param g_a_channel_sizes: list of 4 numbers of channels for parametric transform :math:`g_a`
    :type g_a_channel_sizes: list[int] or None
    :param g_s_channel_sizes: list of 4 numbers of channels for parametric transform :math:`g_s`
    :type g_s_channel_sizes: list[int] or None
    """
    def __init__(self, num_input_channels=3, num_latent_channels=16,
                 num_bottleneck_channels=24, num_target_channels=256,
                 g_a_channel_sizes=None, g_s_channel_sizes=None):
        h_a = nn.Sequential(
            nn.Conv2d(num_bottleneck_channels, num_latent_channels, kernel_size=5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_latent_channels, num_latent_channels, kernel_size=5, stride=2, padding=2, bias=False)
        )

        h_s = nn.Sequential(
            nn.ConvTranspose2d(num_latent_channels, num_latent_channels,
                               kernel_size=5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(num_latent_channels, num_latent_channels * 3 // 2,
                               kernel_size=5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_latent_channels * 3 // 2, num_bottleneck_channels * 2,
                      kernel_size=5, stride=1, padding=0, bias=False)
        )
        super().__init__(num_input_channels=num_input_channels, num_latent_channels=num_latent_channels,
                         num_bottleneck_channels=num_bottleneck_channels, num_target_channels=num_target_channels,
                         h_a=h_a, h_s=h_s, g_a_channel_sizes=g_a_channel_sizes, g_s_channel_sizes=g_s_channel_sizes)

    def encode(self, x, **kwargs):
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_shape = z.size()[-2:]
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indices = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indices, means=means_hat)
        return {'strings': [y_strings, z_strings], 'shape': z_shape}

    def decode(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indices = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indices, means=means_hat)
        return self.g_s(y_hat)

    def _forward2train(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        return self.g_s(y_hat)

    def forward(self, x):
        # if fine-tune or evaluate after "update"
        if self.updated:
            if not self.training:
                encoded_obj = self.encode(x)
                decoded_obj = self.decode(**encoded_obj)
                return decoded_obj

            y = self.g_a(x)
            z = self.h_a(y)
            z_hat = self.entropy_bottleneck.dequantize(
                self.entropy_bottleneck.quantize(z, 'dequantize', self._get_means(z))
            )
            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat = self.gaussian_conditional.dequantize(
                self.gaussian_conditional.quantize(y, 'dequantize', means_hat)
            )
            y_hat = y_hat.detach()
            return self.g_s(y_hat)
        return self._forward2train(x)


def get_layer(cls_or_func_name, **kwargs):
    """
    Gets a layer module.

    :param cls_or_func_name: layer class or function name
    :type cls_or_func_name: str
    :param kwargs: kwargs for the layer class or function to build a layer
    :type kwargs: dict
    :return: layer module
    :rtype: nn.Module or None
    """
    if cls_or_func_name in LAYER_CLASS_DICT:
        return LAYER_CLASS_DICT[cls_or_func_name](**kwargs)
    elif cls_or_func_name in LAYER_FUNC_DICT:
        return LAYER_FUNC_DICT[cls_or_func_name](**kwargs)
    return None
