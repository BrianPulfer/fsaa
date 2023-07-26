from torch.nn import Module

from fsaa.core import (PerceptualMask, PerturbationInitializer,
                       PerturbationUpdater)
from fsaa.initializers.random import RandomInitializer
from fsaa.initializers.random_sign import RandomSignInitializer
from fsaa.losses.cossim_loss import CosSimLoss
from fsaa.losses.lpips_loss import LPIPSAlexLoss, LPIPSVGGLoss
from fsaa.losses.mse_loss import MeanSquaredErrorLoss
from fsaa.masks.custom import CustomMask
from fsaa.masks.jnd import JNDMask
from fsaa.masks.nomask import NoMask
from fsaa.models.cae.cae import SUPPORTED_CAE_MODELS, CAEModel
from fsaa.models.hf.hf_models import SUPPORTED_HF_MODELS, HFModel
from fsaa.models.hub.hub_models import SUPPORTED_HUB_MODELS, HubModel
from fsaa.models.ibot.ibot import SUPPORTED_IBOT_MODELS, iBOTModel
from fsaa.models.ijepa.ijepa import IJEPA, SUPPORTED_IJEPA_MODELS
from fsaa.updaters.fgsm import FGSMUpdater
from fsaa.updaters.langevin import LangevinUpdater
from fsaa.updaters.pgd import PGDUpdater
from fsaa.updaters.random import RandomUpdater

INITIALIZERS = {
    "Random": RandomInitializer,
    "RandomSign": RandomSignInitializer,
}

UPDATERS = {
    "FGSM": FGSMUpdater,
    "Langevin": LangevinUpdater,
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
    "JND": JNDMask,
    "Custom": CustomMask,
    "NoMask": NoMask
}

SUPPORTED_MODELS = (
    SUPPORTED_CAE_MODELS +
    SUPPORTED_HUB_MODELS +
    SUPPORTED_HF_MODELS +
    SUPPORTED_IBOT_MODELS +
    SUPPORTED_IJEPA_MODELS
)


def get_initializer(name: str, lr: float, **kwargs) -> PerturbationInitializer:
    """Returns the initializer used for the attack."""
    return INITIALIZERS[name](lr, **kwargs)


def get_updater(name: str, lr: float, **kwargs) -> PerturbationUpdater:
    """Returns the updater used for the attack."""
    return UPDATERS[name](lr, **kwargs)


def get_loss(name: str, **kwargs) -> Module:
    """Returns the loss used for the attack."""
    return LOSSES[name](**kwargs)


def get_mask(name: str, **kwargs) -> PerceptualMask:
    """Returns the mask used for the attack."""
    return MASKS[name](**kwargs)


def get_model(model_name: str) -> Module:
    """Returns the model used for the attack."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not supported. \
                Pick one of {SUPPORTED_MODELS}"
        )

    if model_name in SUPPORTED_CAE_MODELS:
        return CAEModel(model_name)

    if model_name in SUPPORTED_HUB_MODELS:
        return HubModel(model_name)

    if model_name in SUPPORTED_HF_MODELS:
        return HFModel(model_name)

    if model_name in SUPPORTED_IBOT_MODELS:
        return iBOTModel(model_name)

    if model_name in SUPPORTED_IJEPA_MODELS:
        return IJEPA(model_name)
