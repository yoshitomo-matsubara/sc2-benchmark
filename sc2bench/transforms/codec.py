import os
from io import BytesIO
from tempfile import mkstemp

from PIL import Image
from compressai.utils.bench.codecs import run_command
from torch import nn
from torchdistill.datasets.transform import register_transform_class
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

CODEC_MODULE_DICT = dict()


def register_codec_module(cls):
    CODEC_MODULE_DICT[cls.__name__] = cls
    register_transform_class(cls)
    return cls


@register_codec_module
class WrappedResize(Resize):
    """
    `Resize` in torchvision wrapped to be defined by `interpolation` as a str object
    Args:
        interpolation (str or None): Desired interpolation mode (`nearest`, `bicubic`, `bilinear`, `box`, `hamming`, `lanczos`)
        kwargs (dict): kwargs for `Resize` in torchvision
    """
    MODE_DICT = {
        'nearest': InterpolationMode.NEAREST,
        'bicubic': InterpolationMode.BICUBIC,
        'bilinear': InterpolationMode.BILINEAR,
        'box': InterpolationMode.BOX,
        'hamming': InterpolationMode.HAMMING,
        'lanczos': InterpolationMode.LANCZOS
    }

    def __init__(self, interpolation=None, **kwargs):
        if interpolation is not None:
            interpolation = self.MODE_DICT.get(interpolation, None)
        super().__init__(**kwargs, interpolation=interpolation)


@register_codec_module
class PillowImageModule(nn.Module):
    """
    Generalized Pillow module to compress (decompress) images e.g., as part of transform pipeline.
    Args:
        open_kwargs (dict or None): kwargs to be used as part of Image.open(img_buffer, **open_kwargs).
        save_kwargs (dict or None): kwargs to be used as part of Image.save(img_buffer, **save_kwargs).
    """
    def __init__(self, open_kwargs=None, **save_kwargs):
        super().__init__()
        self.open_kwargs = open_kwargs if isinstance(open_kwargs, dict) else dict()
        self.save_kwargs = save_kwargs

    def forward(self, pil_img):
        """
        Args:
            pil_img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        img_buffer = BytesIO()
        pil_img.save(img_buffer, **self.save_kwargs)
        return Image.open(img_buffer, **self.open_kwargs)


@register_codec_module
class BpgModule(nn.Module):
    """
    Generalized Pillow module to compress (decompress) images e.g., as part of transform pipeline.
    Modified https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/utils/bench/codecs.py
    Args:
        encoder_path (str): file path of BPG encoder you manually installed.
        decoder_path (str): file path of BPG decoder you manually installed.
        color_mode (str): color mode ("ycbcr" or "rgb").
        encoder (str): encoder type ("x265" or "jctvc").
        subsampling_mode (str or int): subsampling mode (420 or 444).
        bit_depth (str or int): bit depth (8 or 10).
        quality (int): quality value in range [0, 51].
    """
    fmt = '.bpg'

    def __init__(self, encoder_path, decoder_path, color_mode='ycbcr', encoder='x265',
                 subsampling_mode='444', bit_depth='8', quality=50):
        super().__init__()
        if not isinstance(subsampling_mode, str):
            subsampling_mode = str(subsampling_mode)

        if not isinstance(bit_depth, str):
            bit_depth = str(bit_depth)

        if color_mode not in ['ycbcr', 'rgb']:
            raise ValueError(f'Invalid color mode value: `{color_mode}`, which should be either "ycbcr" or "rgb"')

        if encoder not in ['x265', 'jctvc']:
            raise ValueError(f'Invalid encoder value: `{encoder}`, which should be either "x265" or "jctvc"')

        if subsampling_mode not in ['420', '444']:
            raise ValueError(f'Invalid subsampling mode value: `{subsampling_mode}`, which should be either 420 or 444')

        if bit_depth not in ['8', '10']:
            raise ValueError(f'Invalid bit depth value: `{bit_depth}`, which should be either 8 or 10')

        if not 0 <= quality <= 51:
            raise ValueError(f'Invalid quality value: `{quality}`, which should be between 0 and 51')

        self.encoder_path = os.path.expanduser(encoder_path)
        self.decoder_path = os.path.expanduser(decoder_path)
        self.color_mode = color_mode
        self.encoder = encoder
        self.subsampling_mode = subsampling_mode
        self.bit_depth = bit_depth
        self.quality = quality

    def _get_encode_cmd(self, img_file_path, output_file_path):
        cmd = [
            self.encoder_path,
            '-o',
            output_file_path,
            '-q',
            str(self.quality),
            '-f',
            self.subsampling_mode,
            '-e',
            self.encoder,
            '-c',
            self.color_mode,
            '-b',
            self.bit_depth,
            img_file_path
        ]
        return cmd

    def _get_decode_cmd(self, out_filepath, reconst_file_path):
        cmd = [self.decoder_path, '-o', reconst_file_path, out_filepath]
        return cmd

    def run(self, pil_img):
        """
        Args:
            pil_img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        fd_i, resized_input_filepath = mkstemp(suffix='.jpg')
        fd_r, reconst_file_path = mkstemp(suffix='.jpg')
        fd_o, output_file_path = mkstemp(suffix=self.fmt)
        pil_img.save(resized_input_filepath, 'JPEG', quality=100)

        # Encode
        run_command(self._get_encode_cmd(resized_input_filepath, output_file_path))

        # Decode
        run_command(self._get_decode_cmd(output_file_path, reconst_file_path))

        # Read image
        reconst_img = Image.open(reconst_file_path).convert('RGB')
        os.close(fd_i)
        os.remove(resized_input_filepath)
        os.close(fd_r)
        os.remove(reconst_file_path)
        os.close(fd_o)
        os.remove(output_file_path)
        return reconst_img
