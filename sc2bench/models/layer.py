import torch
from compressai.entropy_models import GaussianConditional
from compressai.layers import GDN1
from compressai.models import CompressionModel
from compressai.models.google import get_scale_table
from compressai.models.utils import update_registered_buffers
from torch import nn
from torchdistill.common.constant import def_logger
from torchdistill.datasets.util import build_transform

logger = def_logger.getChild(__name__)
LAYER_CLASS_DICT = dict()
LAYER_FUNC_DICT = dict()


def register_layer_class(cls):
    """
    Args:
        cls (class): layer module to be registered.

    Returns:
        cls (class): registered layer module.
    """
    LAYER_CLASS_DICT[cls.__name__] = cls
    return cls


def register_layer_func(func):
    """
    Args:
        func (function): layer module to be registered.

    Returns:
        func (function): registered layer module.
    """
    LAYER_FUNC_DICT[func.__name__] = func
    return func


class SimpleBottleneck(nn.Module):
    """
    Simple encoder-decoder layer to treat encoder's output as bottleneck
    """
    def __init__(self, encoder, decoder, compressor=None, decompressor=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.compressor = compressor
        self.decompressor = decompressor

    def encode(self, x):
        z = self.encoder(x)
        if self.compressor is not None:
            z = self.compressor(z)
        return {'z': z}

    def decode(self, z):
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
        logger.info('This module has no updatable parameters for entropy coding')


@register_layer_func
def larger_resnet_bottleneck(bottleneck_channel=12, bottleneck_idx=12, output_channel=256,
                             compressor_transform_params=None, decompressor_transform_params=None):
    """
    "Head Network Distillation: Splitting Distilled Deep Neural Networks for Resource-constrained Edge Computing Systems"
    """
    modules = [
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False),
        nn.BatchNorm2d(bottleneck_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 128, kernel_size=2, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, output_channel, kernel_size=2, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.Conv2d(output_channel, output_channel, kernel_size=2, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(inplace=True)
    ]
    encoder = nn.Sequential(*modules[:bottleneck_idx])
    decoder = nn.Sequential(*modules[bottleneck_idx:])
    compressor_transform = build_transform(compressor_transform_params)
    decompressor_transform = build_transform(decompressor_transform_params)
    return SimpleBottleneck(encoder, decoder, compressor_transform, decompressor_transform)


class EntropyBottleneckLayer(CompressionModel):
    """
    Entropy bottleneck layer as a simple CompressionModel in compressai
    The entropy bottleneck layer is proposed in "Variational Image Compression with a Scale Hyperprior" by
    J. Balle, D. Minnen, S. Singh, S.J. Hwang, N. Johnston.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.updated = False

    def forward(self, x):
        return self.entropy_bottleneck(x)

    def compress(self, x):
        strings = self.entropy_bottleneck.compress(x)
        return {'strings': [strings], 'shape': x.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        return self.entropy_bottleneck.decompress(strings[0], shape)

    def update(self, force=False):
        self.updated = True
        return super().update(force=force)


class BaseBottleneck(CompressionModel):
    def __init__(self, entropy_bottleneck_channels):
        super().__init__(entropy_bottleneck_channels=entropy_bottleneck_channels)
        self.updated = False

    def encode(self, *args, **kwargs):
        raise NotImplementedError()

    def decode(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args):
        raise NotImplementedError()

    def update(self, force=False):
        self.updated = True
        return super().update(force=force)


@register_layer_class
class FPBasedResNetBottleneck(BaseBottleneck):
    """
    Factorized Prior(FP)-based bottleneck for ResNet proposed in
    "Supervised Compression for Resource-Constrained Edge Computing Systems"
    by Y. Matsubara, R. Yang, M. Levorato, S. Mandt.
    Factorized Prior is proposed in "Variational Image Compression with a Scale Hyperprior" by
    J. Balle, D. Minnen, S. Singh, S.J. Hwang, N. Johnston.
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
        latent = self.encoder(x)
        latent_strings = self.entropy_bottleneck.compress(latent)
        return {'strings': [latent_strings], 'shape': latent.size()[-2:]}

    def decode(self, strings, shape):
        latent_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        return self.decoder(latent_hat)

    def get_means(self, x):
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        return medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))

    def forward2train(self, x):
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
                    self.entropy_bottleneck.quantize(encoded_output, 'dequantize', self.get_means(encoded_output))
                )
            decoder_input = decoder_input.detach()
            return self.decoder(decoder_input)
        return self.forward2train(x)


@register_layer_class
class SHPBasedResNetBottleneck(BaseBottleneck):
    """
    Scale Hyperprior(SHP)-based bottleneck for ResNet.
    Scale Hyperprior is proposed in "Variational Image Compression with a Scale Hyperprior" by
    J. Balle, D. Minnen, S. Singh, S.J. Hwang, N. Johnston.
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
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indices = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indices, z_hat.dtype)
        return self.g_s(y_hat)

    def get_means(self, x):
        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        return medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))

    def forward2train(self, x):
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
                self.gaussian_conditional.quantize(y, 'dequantize', self.get_means(y))
            )
            y_hat = y_hat.detach()
            return self.g_s(y_hat)
        return self.forward2train(x)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()

        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        self.updated = True
        return updated

    def load_state_dict(self, state_dict, **kwargs):
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
    Mean-Scale Hyperprior(MSHP)-based bottleneck for ResNet.
    Mean-Scale Hyperprior is proposed in "Joint Autoregressive and Hierarchical Priors for Learned Image Compression" by
    D. Minnen, J. Balle, G.D. Toderici.
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

    def forward2train(self, x):
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
                self.entropy_bottleneck.quantize(z, 'dequantize', self.get_means(z))
            )
            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat = self.gaussian_conditional.dequantize(
                self.gaussian_conditional.quantize(y, 'dequantize', means_hat)
            )
            y_hat = y_hat.detach()
            return self.g_s(y_hat)
        return self.forward2train(x)


def get_layer(cls_or_func_name, **kwargs):
    """
    Args:
        cls_or_func_name (str): layer class name.
        kwargs (dict): keyword arguments.

    Returns:
        nn.Module or None: layer module that is instance of `nn.Module` if found. None otherwise.
    """
    if cls_or_func_name in LAYER_CLASS_DICT:
        return LAYER_CLASS_DICT[cls_or_func_name](**kwargs)
    elif cls_or_func_name in LAYER_FUNC_DICT:
        return LAYER_FUNC_DICT[cls_or_func_name](**kwargs)
    return None
