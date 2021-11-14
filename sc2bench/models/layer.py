from compressai.layers import GDN1
from compressai.models import CompressionModel
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
    def __init__(self, num_enc_channels=16, num_target_channels=256):
        super().__init__(entropy_bottleneck_channels=num_enc_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_enc_channels * 4, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels * 4),
            nn.Conv2d(num_enc_channels * 4, num_enc_channels * 2, kernel_size=5, stride=2, padding=2, bias=False),
            GDN1(num_enc_channels * 2),
            nn.Conv2d(num_enc_channels * 2, num_enc_channels, kernel_size=2, stride=1, padding=0, bias=False)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(num_enc_channels, num_target_channels * 2, kernel_size=2, stride=1, padding=1, bias=False),
            GDN1(num_target_channels * 2, inverse=True),
            nn.Conv2d(num_target_channels * 2, num_target_channels, kernel_size=2, stride=1, padding=0, bias=False),
            GDN1(num_target_channels, inverse=True),
            nn.Conv2d(num_target_channels, num_target_channels, kernel_size=2, stride=1, padding=1, bias=False)
        )

    def encode(self, x, **kwargs):
        latent = self.encoder(x)
        latent_strings = self.entropy_bottleneck.compress(latent)
        return {'strings': [latent_strings], 'shape': latent.size()[-2:]}

    def decode(self, compressed_obj, **kwargs):
        compressed_latent, latent_shape = compressed_obj
        latent_hat = self.entropy_bottleneck.decompress(compressed_latent, latent_shape)
        return self.decoder(latent_hat)

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

        encoded_obj = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(encoded_obj)
        decoded_obj = self.decoder(y_hat)
        return decoded_obj


def get_layer(cls_name, **kwargs):
    if cls_name not in LAYER_CLASS_DICT:
        return None
    return LAYER_CLASS_DICT[cls_name](**kwargs)
