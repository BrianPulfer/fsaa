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
from fsaa.updaters.fgsm import FGSMUpdater
from fsaa.updaters.langevin import LangevinUpdater
from fsaa.updaters.pgd import PGDUpdater
from fsaa.updaters.random import RandomUpdater

INITIALIZERS = {
    "RandomInitializer": RandomInitializer,
    "RandomSignInitializer": RandomSignInitializer,
}

UPDATERS = {
    "FGSMUpdater": FGSMUpdater,
    "LangevinUpdater": LangevinUpdater,
    "PGDUpdater": PGDUpdater,
    "RandomUpdater": RandomUpdater,
}

LOSSES = {
    "CosSimLoss": CosSimLoss,
    "MeanSquaredErrorLoss": MeanSquaredErrorLoss,
    "LPIPSAlexLoss": LPIPSAlexLoss,
    "LPIPSVGGLoss": LPIPSVGGLoss,
}

MASKS = {
    "JNDMask": JNDMask,
    "CustomMask": CustomMask,
}


def get_initializer(
        name: str,
        alpha: float,
        **kwargs) -> PerturbationInitializer:
    """Returns the initializer used for the attack."""
    return INITIALIZERS[name](alpha, **kwargs)


def get_updater(
        name: str,
        alpha: float,
        **kwargs
) -> PerturbationUpdater:
    """Returns the updater used for the attack."""
    return UPDATERS[name](alpha, **kwargs)


def get_loss(name: str, **kwargs) -> Module:
    """Returns the loss used for the attack."""
    return LOSSES[name](**kwargs)


def get_mask(name: str, **kwargs) -> PerceptualMask:
    """Returns the mask used for the attack."""
    return MASKS[name](**kwargs)
