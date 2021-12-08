import os
from io import BytesIO
from tempfile import mkstemp

import numpy as np
import torch
from PIL import Image
from compressai.transforms.functional import rgb2ycbcr, ycbcr2rgb
from compressai.utils.bench.codecs import run_command
from torch import nn
from torchdistill.common import file_util
from torchdistill.datasets.transform import register_transform_class
from torchvision.transforms import RandomResizedCrop, Resize
from torchvision.transforms.functional import InterpolationMode, to_pil_image, to_tensor

CODEC_TRANSFORM_MODULE_DICT = dict()
INTERPOLATION_MODE_DICT = {
    'nearest': InterpolationMode.NEAREST,
    'bicubic': InterpolationMode.BICUBIC,
    'bilinear': InterpolationMode.BILINEAR,
    'box': InterpolationMode.BOX,
    'hamming': InterpolationMode.HAMMING,
    'lanczos': InterpolationMode.LANCZOS
}


def register_codec_transform_module(cls):
    """
    Args:
        cls (class): codec transform module to be registered.

    Returns:
        cls (class): registered codec transform module.
    """
    CODEC_TRANSFORM_MODULE_DICT[cls.__name__] = cls
    register_transform_class(cls)
    return cls


@register_codec_transform_module
class WrappedRandomResizedCrop(RandomResizedCrop):
    """
    `RandomResizedCrop` in torchvision wrapped to be defined by `interpolation` as a str object
    Args:
        interpolation (str or None): Desired interpolation mode (`nearest`, `bicubic`, `bilinear`, `box`, `hamming`, `lanczos`)
        kwargs (dict): kwargs for `RandomResizedCrop` in torchvision
    """
    def __init__(self, interpolation=None, **kwargs):
        if interpolation is not None:
            interpolation = INTERPOLATION_MODE_DICT.get(interpolation, None)
        super().__init__(**kwargs, interpolation=interpolation)


@register_codec_transform_module
class WrappedResize(Resize):
    """
    `Resize` in torchvision wrapped to be defined by `interpolation` as a str object
    Args:
        interpolation (str or None): Desired interpolation mode (`nearest`, `bicubic`, `bilinear`, `box`, `hamming`, `lanczos`)
        kwargs (dict): kwargs for `Resize` in torchvision
    """
    def __init__(self, interpolation=None, **kwargs):
        if interpolation is not None:
            interpolation = INTERPOLATION_MODE_DICT.get(interpolation, None)
        super().__init__(**kwargs, interpolation=interpolation)


@register_codec_transform_module
class PillowImageModule(nn.Module):
    """
    Generalized Pillow module to compress (decompress) images e.g., as part of transform pipeline.
    Args:
        returns_file_size (bool): return file size of compressed object in addition to PIL image if true.
        open_kwargs (dict or None): kwargs to be used as part of Image.open(img_buffer, **open_kwargs).
        save_kwargs (dict or None): kwargs to be used as part of Image.save(img_buffer, **save_kwargs).
    """
    def __init__(self, returns_file_size=False, open_kwargs=None, **save_kwargs):
        super().__init__()
        self.returns_file_size = returns_file_size
        self.open_kwargs = open_kwargs if isinstance(open_kwargs, dict) else dict()
        self.save_kwargs = save_kwargs

    def forward(self, pil_img, *args):
        """
        Args:
            pil_img (PIL Image): Image to be transformed.

        Returns:
            PIL Image or a tuple of PIL Image and int: Affine transformed image or with its file size if returns_file_size=True.
        """
        img_buffer = BytesIO()
        pil_img.save(img_buffer, **self.save_kwargs)
        file_size = img_buffer.tell()
        pil_img = Image.open(img_buffer, **self.open_kwargs)
        if self.returns_file_size:
            return pil_img, file_size
        return pil_img

    def __repr__(self):
        return self.__class__.__name__ + \
               '(returns_file_size={}, open_kwargs={}, save_kwargs={})'.format(self.returns_file_size,
                                                                               self.open_kwargs, self.save_kwargs)


@register_codec_transform_module
class PillowTensorModule(nn.Module):
    """
    Generalized Pillow module to compress (decompress) tensors e.g., as part of transform pipeline.
    Args:
        returns_file_size (bool): return file size of compressed object in addition to PIL image if true.
        open_kwargs (dict or None): kwargs to be used as part of Image.open(img_buffer, **open_kwargs).
        save_kwargs (dict or None): kwargs to be used as part of Image.save(img_buffer, **save_kwargs).
    """
    def __init__(self, returns_file_size=False, open_kwargs=None, **save_kwargs):
        super().__init__()
        self.returns_file_size = returns_file_size
        self.open_kwargs = open_kwargs if isinstance(open_kwargs, dict) else dict()
        self.save_kwargs = save_kwargs

    def forward(self, x, *args):
        """
        Args:
            x (torch.Tensor): Tensor (C, H, W) to be transformed

        Returns:
            torch.Tensor or a tuple of torch.Tensor and int: Affine transformed image or with its file size if returns_file_size=True.
        """
        device = x.device
        split_features = x.split(3, dim=0)
        last_shape = split_features[-1].shape
        if last_shape[0] == 2:
            more_split_last_features = split_features[-1].split(1, dim=0)
            split_features = split_features[:-1] + more_split_last_features

        file_size = 0
        norm_max_list, norm_min_list, reconstructed_split_feature_list = list(), list(), list()
        for split_feature in split_features:
            # split_feature: (3 or 1, H, W)
            max_value = split_feature.max()
            min_value = split_feature.min()
            norm_max_list.append(max_value)
            norm_min_list.append(min_value)
            # normalize to [0, 1]
            normed_feature = (split_feature - min_value) / max_value
            pil_img = to_pil_image(normed_feature)
            img_buffer = BytesIO()
            # Compress split feature by codec
            pil_img.save(img_buffer, **self.save_kwargs)
            file_size += img_buffer.tell()
            pil_img = Image.open(img_buffer, **self.open_kwargs)
            if split_feature.shape[0] == 1 and pil_img.mode != 'L':
                pil_img = pil_img.convert('L')

            tensor = to_tensor(pil_img)
            tensor = tensor.to(device) * max_value + min_value
            reconstructed_split_feature_list.append(tensor)

        reconstructed_features = torch.vstack(reconstructed_split_feature_list)
        # File size: Compressed feature by codec + values to denormalize (norm_min_list, norm_max_list)
        norm_data_size = \
            file_util.get_binary_object_size(norm_min_list, unit_size=1) \
            + file_util.get_binary_object_size(norm_max_list, unit_size=1)
        file_size += norm_data_size
        if self.returns_file_size:
            return reconstructed_features, file_size
        return reconstructed_features

    def __repr__(self):
        return self.__class__.__name__ + \
               '(returns_file_size={}, open_kwargs={}, save_kwargs={})'.format(self.returns_file_size,
                                                                               self.open_kwargs, self.save_kwargs)


@register_codec_transform_module
class BPGModule(nn.Module):
    """
    BPG module to compress (decompress) images e.g., as part of transform pipeline.
    Modified https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/utils/bench/codecs.py
    Args:
        encoder_path (str): file path of BPG encoder you manually installed.
        decoder_path (str): file path of BPG decoder you manually installed.
        color_mode (str): color mode ("ycbcr" or "rgb").
        encoder (str): encoder type ("x265" or "jctvc").
        subsampling_mode (str or int): subsampling mode (420 or 444).
        bit_depth (str or int): bit depth (8 or 10).
        quality (int): quality value in range [0, 51].
        returns_file_size (bool): flag to return file size.
    """

    fmt = '.bpg'

    def __init__(self, encoder_path, decoder_path, color_mode='ycbcr', encoder='x265',
                 subsampling_mode='444', bit_depth='8', quality=50, returns_file_size=False):
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
        self.returns_file_size = returns_file_size

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

    def _get_decode_cmd(self, output_file_path, reconst_file_path):
        cmd = [self.decoder_path, '-o', reconst_file_path, output_file_path]
        return cmd

    def forward(self, pil_img):
        """
        Args:
            pil_img (PIL Image): Image to be transformed.

        Returns:
            PIL Image or a tuple of PIL Image and float: Affine transformed image or with its file size of BPG compressed data if returns_file_size=True.
        """
        fd_i, resized_input_filepath = mkstemp(suffix='.jpg')
        fd_r, reconst_file_path = mkstemp(suffix='.jpg')
        fd_o, output_file_path = mkstemp(suffix=self.fmt)
        pil_img.save(resized_input_filepath, 'JPEG', quality=100)

        # Encode
        run_command(self._get_encode_cmd(resized_input_filepath, output_file_path))
        file_size_byte = os.stat(output_file_path).st_size

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
        if self.returns_file_size:
            return reconst_img, file_size_byte
        return reconst_img

    def __repr__(self):
        return self.__class__.__name__ + '(encoder_path={}, decoder_path={}, color_mode={}, ' \
                                         'encoder={}, subsampling_mode={}, bit_depth={}, quality={}, ' \
                                         'returns_file_size={})'.format(self.encoder_path, self.decoder_path,
                                                                        self.color_mode, self.encoder,
                                                                        self.subsampling_mode, self.bit_depth,
                                                                        self.quality, self.returns_file_size)


@register_codec_transform_module
class VTMModule(nn.Module):
    """
    VTM module to compress (decompress) images e.g., as part of transform pipeline.
    Modified https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/utils/bench/codecs.py
    Args:
        encoder_path (str): file path of BPG encoder you manually installed.
        decoder_path (str): file path of BPG decoder you manually installed.
        config_path (str): VTM configuration file path.
        color_mode (str): color mode ("ycbcr" or "rgb").
        quality (int): quality value in range [0, 63].
        returns_file_size (bool): flag to return file size.
    """

    fmt = '.bin'

    def __init__(self, encoder_path, decoder_path, config_path, color_mode='ycbcr',
                 quality=63, returns_file_size=False):
        # According to https://github.com/InterDigitalInc/CompressAI/issues/31,
        # CompressAI used "encoder_intra_vtm.cfg" config file
        super().__init__()
        if color_mode not in ['ycbcr', 'rgb']:
            raise ValueError(f'Invalid color mode value: `{color_mode}`, which should be either "ycbcr" or "rgb"')

        if not 0 <= quality <= 63:
            raise ValueError(f'Invalid quality value: `{quality}`, which should be between 0 and 63')

        self.encoder_path = os.path.expanduser(encoder_path)
        self.decoder_path = os.path.expanduser(decoder_path)
        self.config_path = os.path.expanduser(config_path)
        self.uses_rgb = color_mode != 'ycbcr'
        self.quality = quality
        self.returns_file_size = returns_file_size

    def forward(self, pil_img):
        """
        Args:
            pil_img (PIL Image): Image to be transformed.

        Returns:
            PIL Image or a tuple of PIL Image and float: Affine transformed image or with its file size of VTM compressed data if returns_file_size=True.
        """

        # Taking 8bit input for now
        bitdepth = 8

        # Convert input image to yuv 444 file
        arr = np.asarray(pil_img)
        fd, yuv_path = mkstemp(suffix='.yuv')
        output_file_path = os.path.splitext(yuv_path)[0] + self.fmt
        arr = arr.transpose((2, 0, 1))  # color channel first

        if not self.uses_rgb:
            # convert rgb content to YCbCr
            rgb = torch.from_numpy(arr.copy()).float() / (2 ** bitdepth - 1)
            arr = np.clip(rgb2ycbcr(rgb).numpy(), 0, 1)
            arr = (arr * (2 ** bitdepth - 1)).astype(np.uint8)

        with open(yuv_path, 'wb') as f:
            f.write(arr.tobytes())

        # Encode
        height, width = arr.shape[1:]
        cmd = [
            self.encoder_path,
            '-i',
            yuv_path,
            '-c',
            self.config_path,
            '-q',
            self.quality,
            '-o',
            '/dev/null',
            '-b',
            output_file_path,
            '-wdt',
            width,
            '-hgt',
            height,
            '-fr',
            '1',
            '-f',
            '1',
            '--InputChromaFormat=444',
            '--InputBitDepth=8',
            # "--ConformanceMode=1" # It looks like ConformanceMode arg is no longer supported
        ]

        if self.uses_rgb:
            cmd += [
                '--InputColourSpaceConvert=RGBtoGBR',
                '--SNRInternalColourSpace=1',
                '--OutputInternalColourSpace=0',
            ]
        run_command(cmd)
        file_size = os.stat(output_file_path).st_size
        # cleanup encoder input
        os.close(fd)
        os.unlink(yuv_path)

        # Decode
        cmd = [self.decoder_path, '-b', output_file_path, '-o', yuv_path, '-d', 8]
        if self.uses_rgb:
            cmd.append('--OutputInternalColourSpace=GBRtoRGB')

        run_command(cmd)

        # Compute PSNR
        rec_arr = np.fromfile(yuv_path, dtype=np.uint8)
        rec_arr = rec_arr.reshape(arr.shape)

        rec_arr = rec_arr.astype(np.float32) / (2 ** bitdepth - 1)
        if not self.uses_rgb:
            rec_arr = ycbcr2rgb(torch.from_numpy(rec_arr.copy())).numpy()

        # Cleanup
        os.unlink(yuv_path)
        os.unlink(output_file_path)
        rec = Image.fromarray((rec_arr.clip(0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8))

        if self.returns_file_size:
            return rec, file_size
        return rec

    def __repr__(self):
        return self.__class__.__name__ + '(encoder_path={}, decoder_path={}, config_path={}, ' \
                                         'uses_rgb={}, quality={}, ' \
                                         'returns_file_size={})'.format(self.encoder_path, self.decoder_path,
                                                                        self.config_path, self.uses_rgb, self.quality,
                                                                        self.returns_file_size)
