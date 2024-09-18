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


def register_compressai_model(cls_or_func):
    """
    Registers a compression model class or a function to build a compression model in `compressai`.

    :param cls_or_func: compression model or function to build a compression model to be registered
    :type cls_or_func: class or typing.Callable
    :return: registered compression model class or function
    :rtype: class or typing.Callable
    """
    COMPRESSAI_DICT[cls_or_func.__name__] = cls_or_func
    return cls_or_func


def register_compression_model_class(cls):
    """
    Registers a compression model class.

    :param cls: compression model to be registered
    :type cls: class
    :return: registered compression model class
    :rtype: class
    """
    COMPRESSION_MODEL_CLASS_DICT[cls.__name__] = cls
    return cls


def register_compression_model_func(func):
    """
    Registers a function to build a compression model.

    :param func: function to build a compression model to be registered
    :type func: typing.Callable
    :return: registered function
    :rtype: typing.Callable
    """
    COMPRESSION_MODEL_FUNC_DICT[func.__name__] = func
    return func


def get_compressai_model(compression_model_name, ckpt_file_path=None, updates=False, **compression_model_kwargs):
    """
    Gets a model in `compressai`.

    :param compression_model_name: `compressai` model name
    :type compression_model_name: str
    :param ckpt_file_path: checkpoint file path
    :type ckpt_file_path: str or None
    :param updates: if True, updates the parameters for entropy coding
    :type updates: bool
    :param compression_model_kwargs: kwargs for the model class or function to build the model
    :type compression_model_kwargs: dict
    :return: `compressai` model
    :rtype: nn.Module
    """
    compression_model = COMPRESSAI_DICT[compression_model_name](**compression_model_kwargs)
    if ckpt_file_path is not None:
        load_ckpt(ckpt_file_path, model=compression_model, strict=None)

    if updates:
        logger.info('Updating compression model')
        compression_model.update()
    return compression_model


def get_compression_model(compression_model_config, device):
    """
    Gets a compression model.

    :param compression_model_config: compression model configuration
    :type compression_model_config: dict or None
    :param device: torch device
    :type device: str or torch.device
    :return: compression model
    :rtype: nn.Module
    """
    if compression_model_config is None:
        return None

    compression_model_name = compression_model_config['name']
    compression_model_kwargs = compression_model_config['params']
    compression_model_ckpt_file_path = compression_model_config.get('src_ckpt', None)
    if compression_model_name in COMPRESSAI_DICT:
        compression_model_update = compression_model_config.get('update', True)
        compression_model = get_compressai_model(compression_model_name, compression_model_ckpt_file_path,
                                                 compression_model_update, **compression_model_kwargs)
        return compression_model.to(device)
    raise ValueError('compression_model_name `{}` is not expected'.format(compression_model_name))


def load_classification_model(model_config, device, distributed, strict=True):
    """
    Loads an image classification model.

    :param model_config: image classification model configuration
    :type model_config: dict
    :param device: torch device
    :type device: str or torch.device
    :param distributed: whether to use the model in distributed training mode
    :type distributed: bool
    :param strict: whether to strictly enforce that the keys in state_dict match the keys returned by the model’s
            `state_dict()` function
    :type strict: bool
    :return: image classification model
    :rtype: nn.Module
    """
    model = get_image_classification_model(model_config, distributed)
    model_name = model_config['name']
    if model is None and model_name in timm.models.__dict__:
        model = timm.models.__dict__[model_name](**model_config['params'])

    if model is None:
        model = get_backbone(model_name, **model_config['params'])

    if model is None:
        repo_or_dir = model_config.get('repo_or_dir', None)
        model = get_model(model_name, repo_or_dir, **model_config['params'])

    src_ckpt_file_path = model_config.get('src_ckpt', None)
    if src_ckpt_file_path is not None:
        load_ckpt(src_ckpt_file_path, model=model, strict=strict)
    return model.to(device)
