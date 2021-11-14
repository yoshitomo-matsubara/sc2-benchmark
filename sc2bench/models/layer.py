import torch
from compressai.entropy_models import GaussianConditional
from compressai.layers import GDN1
from compressai.models import CompressionModel
from compressai.models.priors import get_scale_table
from compressai.models.utils import update_registered_buffers
from torch import nn

LAYER_CLASS_DICT = dict()


def register_layer_class(cls):
    """
    Args:
        cls (class): layer module to be registered.

    Returns:
        cls (class): registered layer module.
    """
    LAYER_CLASS_DICT[cls.__name__] = cls
    return cls


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
        super().update(force=force)
        self.updated = True


@register_layer_class
class FPBasedResNetBottleneck(BaseBottleneck):
    """
    Factorized Prior(FP)-based bottleneck for ResNet proposed in
    "Supervised Compression for Resource-Constrained Edge Computing Systems"
    by Y. Matsubara, R. Yang, M. Levorato, S. Mandt.
    Factorized Prior is proposed in "Variational Image Compression with a Scale Hyperprior" by
    J. Balle, D. Minnen, S. Singh, S.J. Hwang, N. Johnston.
    """
    def __init__(self, num_input_channels=3, num_bottleneck_channels=16, num_target_channels=256):
        super().__init__(entropy_bottleneck_channels=num_bottleneck_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, num_bottleneck_channels * 4, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_bottleneck_channels * 4),
            nn.Conv2d(num_bottleneck_channels * 4, num_bottleneck_channels * 2,
                      kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_bottleneck_channels * 2),
            nn.Conv2d(num_bottleneck_channels * 2, num_bottleneck_channels,
                      kernel_size=2, stride=1, padding=0, bias=False)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(num_bottleneck_channels, num_target_channels * 2, kernel_size=2, stride=1, padding=1, bias=False),
            GDN1(num_target_channels * 2, inverse=True),
            nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=2, stride=1, padding=0, bias=False),
            GDN1(num_target_channels, inverse=True),
            nn.Conv2d(num_target_channels, num_target_channels, kernel_size=2, stride=1, padding=1, bias=False)
        )

    def encode(self, x, **kwargs):
        latent = self.encoder(x)
        latent_strings = self.entropy_bottleneck.compress(latent)
        return {'strings': [latent_strings], 'shape': latent.size()[-2:]}

    def decode(self, strings, shape):
        latent_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        return self.decoder(latent_hat)

    def forward2train(self, x):
        encoded_obj = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(encoded_obj)
        decoded_obj = self.decoder(y_hat)
        return decoded_obj

    def forward(self, x):
        if not self.training:
            encoded_obj = self.encode(x)
            decoded_obj = self.decode(**encoded_obj)
            return decoded_obj

        # if fine-tuning after "update"
        if self.updated:
            encoded_output = self.encoder(x)
            decoder_input =\
                self.entropy_bottleneck.dequantize(self.entropy_bottleneck.quantize(encoded_output, 'dequantize'))
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
    def __init__(self, num_input_channels=3, num_latent_channels=64,
                 num_bottleneck_channels=16, num_target_channels=256, h_a=None, h_s=None):
        super().__init__(entropy_bottleneck_channels=num_latent_channels)
        self.g_a = nn.Sequential(
            nn.Conv2d(num_input_channels, num_bottleneck_channels * 4,
                      kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_bottleneck_channels * 4),
            nn.Conv2d(num_bottleneck_channels * 4, num_bottleneck_channels * 2,
                      kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_bottleneck_channels * 2),
            nn.Conv2d(num_bottleneck_channels * 2, num_bottleneck_channels,
                      kernel_size=2, stride=1, padding=0, bias=False)
        )

        self.g_s = nn.Sequential(
            nn.Conv2d(num_bottleneck_channels, num_target_channels * 2, kernel_size=2, stride=1, padding=1, bias=False),
            GDN1(num_target_channels * 2, inverse=True),
            nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=2, stride=1, padding=0, bias=False),
            GDN1(num_latent_channels, inverse=True),
            nn.Conv2d(num_target_channels, num_target_channels, kernel_size=2, stride=1, padding=1, bias=False)
        )

        self.h_a = nn.Sequential(
            nn.Conv2d(num_bottleneck_channels, num_latent_channels, kernel_size=5, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_latent_channels, num_latent_channels, kernel_size=5, stride=2, padding=2, bias=False)
        ) if h_a is None else h_a

        self.h_s = nn.Sequential(
            nn.Conv2d(num_latent_channels, num_latent_channels, kernel_size=2, stride=1, padding=1, bias=False),
            GDN1(num_latent_channels),
            nn.Conv2d(num_latent_channels, num_latent_channels, kernel_size=2, stride=1, padding=0, bias=False),
            GDN1(num_latent_channels),
            nn.ReLU(inplace=True)
        ) if h_s is None else h_s

        self.gaussian_conditional = GaussianConditional(None)
        self.num_latent_channels = num_latent_channels
        self.num_bottleneck_channels = num_bottleneck_channels

    def encode(self, x, **kwargs):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {'strings': [y_strings, z_strings], 'shape': z.size()[-2:]}

    def decode(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        return self.g_s(y_hat)

    def forward2train(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        return self.g_s(y_hat)

    def forward(self, x):
        if not self.training:
            encoded_obj = self.encode(x)
            decoded_obj = self.decode(**encoded_obj)
            return decoded_obj

        # if fine-tuning after "update"
        if self.updated:
            encoded_output = self.encoder(x)
            decoder_input =\
                self.entropy_bottleneck.dequantize(self.entropy_bottleneck.quantize(encoded_output, 'dequantize'))
            decoder_input = decoder_input.detach()
            return self.decoder(decoder_input)
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
        super().load_state_dict(state_dict, **kwargs)


def get_layer(cls_name, **kwargs):
    if cls_name not in LAYER_CLASS_DICT:
        return None
    return LAYER_CLASS_DICT[cls_name](**kwargs)
