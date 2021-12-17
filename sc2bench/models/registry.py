import timm
from compressai.zoo.image import model_architectures
from torchdistill.common import misc_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import load_ckpt
from torchdistill.models.official import get_image_classification_model
from torchdistill.models.registry import get_model

from .backbone import get_backbone

logger = def_logger.getChild(__name__)
COMPRESSAI_DICT = dict()
COMPRESSAI_DICT.update(model_architectures)
COMPRESSAI_DICT.update(misc_util.get_functions_as_dict('compressai.zoo.image'))
COMPRESSION_MODEL_CLASS_DICT = dict()
COMPRESSION_MODEL_FUNC_DICT = dict()


def register_compressai_model_class(cls_or_func):
    COMPRESSAI_DICT[cls_or_func.__name__] = cls_or_func
    return cls_or_func


def register_compression_model_class(cls):
    COMPRESSION_MODEL_CLASS_DICT[cls.__name__] = cls
    return cls


def register_compression_model_func(func):
    COMPRESSION_MODEL_FUNC_DICT[func.__name__] = func
    return func


def get_compressai_model(compression_model_name, ckpt_file_path=None, updates=False, **compression_model_kwargs):
    compression_model = COMPRESSAI_DICT[compression_model_name](**compression_model_kwargs)
    if ckpt_file_path is not None:
        load_ckpt(ckpt_file_path, model=compression_model, strict=None)

    if updates:
        logger.info('Updating compression model')
        compression_model.update()
    return compression_model


def get_compression_model(compression_model_config, device):
    if compression_model_config is None:
        return None

    compression_model_name = compression_model_config['name']
    compression_model_kwargs = compression_model_config['params']
    uses_cpu = compression_model_config.get('uses_cpu', False)
    compression_model_ckpt_file_path = compression_model_config.get('ckpt', None)
    if compression_model_name in COMPRESSAI_DICT:
        compression_model_update = compression_model_config.get('update', True)
        compression_model = get_compressai_model(compression_model_name, compression_model_ckpt_file_path,
                                                 compression_model_update, **compression_model_kwargs)
        return compression_model.cpu() if uses_cpu else compression_model.to(device)
    raise ValueError('compression_model_name `{}` is not expected'.format(compression_model_name))


def load_classification_model(model_config, device, distributed):
    model = get_image_classification_model(model_config, distributed)
    model_name = model_config['name']
    if model is None and model_name in timm.models.__dict__:
        model = timm.models.__dict__[model_name](**model_config['params'])

    if model is None:
        model = get_backbone(model_name, **model_config['params'])

    if model is None:
        repo_or_dir = model_config.get('repo_or_dir', None)
        model = get_model(model_name, repo_or_dir, **model_config['params'])

    ckpt_file_path = model_config['ckpt']
    load_ckpt(ckpt_file_path, model=model, strict=True)
    return model.to(device)
