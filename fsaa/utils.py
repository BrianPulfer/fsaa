from typing import Callable

from torch import nn
from torch.nn import Module

from fsaa.core import (PerceptualMask, PerturbationInitializer,
                       PerturbationUpdater, Scheduler)
from fsaa.initializers.random import RandomInitializer
from fsaa.initializers.random_sign import RandomSignInitializer
from fsaa.losses.cossim_loss import CosSimLoss
from fsaa.losses.lpips_loss import LPIPSAlexLoss, LPIPSVGGLoss
from fsaa.losses.mse_loss import MeanSquaredErrorLoss
from fsaa.masks.custom import CustomMask
from fsaa.masks.jnd import JNDMask
from fsaa.masks.nomask import NoMask
from fsaa.masks.random_crop import RandomCropMask
from fsaa.models.cae.cae import SUPPORTED_CAE_MODELS, CAEModel
from fsaa.models.hf.hf_models import SUPPORTED_HF_MODELS, HFModel
from fsaa.models.hub.hub_models import SUPPORTED_HUB_MODELS, HubModel
from fsaa.models.ibot.ibot import SUPPORTED_IBOT_MODELS, iBOTModel
from fsaa.models.ijepa.ijepa import IJEPA, SUPPORTED_IJEPA_MODELS
from fsaa.updaters.lr_schedulers.function import FunctionScheduler
from fsaa.updaters.lr_schedulers.linear import LinearScheduler
from fsaa.updaters.pgd import PGDUpdater
from fsaa.updaters.random import RandomUpdater

INITIALIZERS = {
    "Random": RandomInitializer,
    "RandomSign": RandomSignInitializer,
}

SCHEDULERS = {
    "Function": FunctionScheduler,
    "Linear": LinearScheduler,
}

UPDATERS = {
    "PGD": PGDUpdater,
    "Random": RandomUpdater,
}

LOSSES = {
    "CosSim": CosSimLoss,
    "MSE": MeanSquaredErrorLoss,
    "LPIPSAlex": LPIPSAlexLoss,
    "LPIPSVGG": LPIPSVGGLoss,
}

MASKS = {
    "Custom": CustomMask,
    "JND": JNDMask,
    "NoMask": NoMask,
    "RandomCrop": RandomCropMask
}

SUPPORTED_MODELS = (
    SUPPORTED_CAE_MODELS
    + SUPPORTED_HUB_MODELS
    + SUPPORTED_HF_MODELS
    + SUPPORTED_IBOT_MODELS
    + SUPPORTED_IJEPA_MODELS
)


def get_initializer(name: str, lr: float, **kwargs) -> PerturbationInitializer:
    """
    Returns a perturbation initializer with the given name.

    Args:
        name (str): Name of the initializer.
        lr (float): Learning rate (or magnitude) for the update.

    Returns:
        PerturbationInitializer: Perturbation initializer with the given name.
    """
    return INITIALIZERS[name](lr, **kwargs)


def get_scheduler(name: str, base_lr: float, **kwargs) -> Scheduler:
    """
    Returns a scheduler with the given name.

    Args:
        name (str): Name of the scheduler.
        base_lr (float): Base learning rate (or magnitude) for the update.

    Returns:
        Scheduler: Scheduler with the given name.
    """
    return SCHEDULERS[name](base_lr, **kwargs)


def get_updater(name: str, lr: float, **kwargs) -> PerturbationUpdater:
    """
    Returns a perturbation updater with the given name.

    Args:
        name (str): Name of the updater.
        lr (float): Learning rate (or magnitude) for the update.

    Returns:
        PerturbationUpdater: Perturbation updater with the given name.
    """
    return UPDATERS[name](lr, **kwargs)


def get_loss(name: str, **kwargs) -> Callable:
    """
    Returns a loss with the given name.

    Args:
        name (str): Name of the loss.

    Returns:
        Callable: Loss with the given name.
    """
    if name in LOSSES:
        return LOSSES[name](**kwargs)
    if hasattr(nn, name):
        kwargs.update({"reduction": "none"})
        return getattr(nn, name)(**kwargs)
    raise ValueError(
        f"Loss '{name}' is not supported. Pick one of {LOSSES} or use a loss either from torch.nn or torch.nn.functional")


def get_mask(name: str, **kwargs) -> PerceptualMask:
    """
    Returns a PerceptualMask with the given name.

    Args:
        name (str): Name of the masking method.

    Returns:
        PerceptualMask: Masking method with the given name.

    """
    return MASKS[name](**kwargs)


def get_model(model_name: str, *args, **kwargs) -> Module:
    """
    Returns the pre-trained model with the given name.
    The model is wrapped with its corresponding pre-processing differentiable transform.

    Args:
        model_name (str): Name of the model.

    Returns:
        Module: Pre-trained model with the given name.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not supported. \
            Pick one of {SUPPORTED_MODELS}"
        )

    if model_name in SUPPORTED_CAE_MODELS:
        return CAEModel(model_name, *args, **kwargs)

    if model_name in SUPPORTED_HUB_MODELS:
        return HubModel(model_name, *args, **kwargs)

    if model_name in SUPPORTED_HF_MODELS:
        return HFModel(model_name, *args, **kwargs)

    if model_name in SUPPORTED_IBOT_MODELS:
        return iBOTModel(model_name, *args, **kwargs)

    if model_name in SUPPORTED_IJEPA_MODELS:
        return IJEPA(model_name, *args, **kwargs)
