from torchdistill.common.main_util import load_ckpt
from torchdistill.models.official import get_semantic_segmentation_model
from torchdistill.models.registry import get_model
from torchdistill.models.registry import register_model_class, register_model_func

SEGMENTATION_MODEL_CLASS_DICT = dict()
SEGMENTATION_MODEL_FUNC_DICT = dict()


def register_segmentation_model_class(cls):
    SEGMENTATION_MODEL_CLASS_DICT[cls.__name__] = cls
    register_model_class(cls)
    return cls


def register_segmentation_model_func(func):
    SEGMENTATION_MODEL_FUNC_DICT[func.__name__] = func
    register_model_func(func)
    return func


def get_segmentation_model(cls_or_func_name, **kwargs):
    if cls_or_func_name in SEGMENTATION_MODEL_CLASS_DICT:
        return SEGMENTATION_MODEL_CLASS_DICT[cls_or_func_name](**kwargs)
    elif cls_or_func_name in SEGMENTATION_MODEL_FUNC_DICT:
        return SEGMENTATION_MODEL_FUNC_DICT[cls_or_func_name](**kwargs)
    return None


def load_segmentation_model(model_config, device):
    model = get_semantic_segmentation_model(model_config)
    model_name = model_config['name']
    if model is None:
        model = get_segmentation_model(model_name, **model_config['params'])

    if model is None:
        repo_or_dir = model_config.get('repo_or_dir', None)
        model = get_model(model_name, repo_or_dir, **model_config['params'])

    ckpt_file_path = model_config['ckpt']
    load_ckpt(ckpt_file_path, model=model, strict=True)
    return model.to(device)
