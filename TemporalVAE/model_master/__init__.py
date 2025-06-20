from .base import *
# 2023-09-24 15:09:37 for mouse adversarial test, use psupertime dataset to test performance
from TemporalVAE.model_master.supervise_vanilla_vae_regressionClfDecoder_mouse_noAdversarial import *

# 2024-01-08 15:28:05 for finetune set unet shape of loss
from TemporalVAE.model_master.supervise_vanilla_vae_regressionClfDecoder_mouse_noAdversarial_u_s_focusEncoder import *
from TemporalVAE.model_master.RNA_velocity_u_s_Ft_on_s import *
from TemporalVAE.model_master.supervise_vanilla_vae import *
# 2024-04-06 17:14:33 add batch decoder

# Aliases

vae_models = {
    "RNA_velocity_u_s_Ft_on_s": RNA_velocity_u_s_Ft_on_s,
    "SuperviseVanillaVAE": SuperviseVanillaVAE,
    "SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial": SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial,
    "SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_u_s_focusEncoder": SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_u_s_focusEncoder,
}
