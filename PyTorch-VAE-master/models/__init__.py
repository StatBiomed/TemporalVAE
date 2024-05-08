from .base import *
from .vanilla_vae import *
from .gamma_vae import *
from .beta_vae import *
from .wae_mmd import *
from .cvae import *
from .hvae import *
from .vampvae import *
from .iwae import *
from .dfcvae import *
from .mssim_vae import MSSIMVAE
from .fvae import *
from .cat_vae import *
from .joint_vae import *
from .info_vae import *
# from .twostage_vae import *
from .lvae import LVAE
from .logcosh_vae import *
from .swae import *
from .miwae import *
from .vq_vae import *
from .betatc_vae import *
from .dip_vae import *
from .supervise_vanilla_vae import *
from .supervise_vanilla_vae_noClfDecoder import *
from .supervise_vanilla_vae_regressionClfDecoder import *
from .supervise_vanilla_vae_regressionClfDecoder_of_subLatentSpace import *

# with remove-batch-effect decoder
from .supervise_vanilla_vae_with_removeBatchEffectDecoder import *
from .supervise_vanilla_vae_regressionClfDecoder_with_removeBatchEffectDecoder import *
from .supervise_vanilla_vae_regressionClfDecoder_adversarial import *
# 2023-09-24 15:09:37 for mouse adversarial test, use psupertime dataset to test performance
from .supervise_vanilla_vae_regressionClfDecoder_mouse_noAdversarial import *
from .supervise_vanilla_vae_discreteLabel_Joy import *
from .supervise_vanilla_vae_discreteLabel_Joy_constrastive_adversarial import *

# 2024-01-08 15:28:05 for finetune set unet shape of loss
from .supervise_vanilla_vae_regressionClfDecoder_mouse_noAdversarial_u_s_focusDecoder import *
from .supervise_vanilla_vae_regressionClfDecoder_mouse_noAdversarial_u_s_focusEncoder import *
from .supervise_vanilla_vae_regressionClfDecoder_mouse_noAdversarial_u_s_focusEncoder_moreFeatures import *
from .RNA_velocity_u_s_Ft_on_s import *

# 2024-04-06 17:14:33 add batch decoder
from .supervise_vanilla_vae_regressionClfDecoder_mouse_batchDecoder import *

# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE
GumbelVAE = CategoricalVAE

vae_models = {
    "SuperviseVanillaVAE_regressionClfDecoder_mouse_batchDecoder": SuperviseVanillaVAE_regressionClfDecoder_mouse_batchDecoder,
    "RNA_velocity_u_s_Ft_on_s": RNA_velocity_u_s_Ft_on_s,
    "SuperviseVanillaVAE_discreteLabel_Joy_constrastive_adversarial": SuperviseVanillaVAE_discreteLabel_Joy_constrastive_adversarial,
    "SuperviseVanillaVAE_discreteLabel_Joy": SuperviseVanillaVAE_discreteLabel_Joy,
    "SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial": SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial,
    "SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_u_s_focusDecoder": SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_u_s_focusDecoder,
    "SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_u_s_focusEncoder": SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_u_s_focusEncoder,
    "SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_u_s_focusEncoder_moreFeatures": SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_u_s_focusEncoder_moreFeatures,
    "SuperviseVanillaVAE_regressionClfDecoder_adversarial": SuperviseVanillaVAE_regressionClfDecoder_adversarial,
    "SuperviseVanillaVAE_regressionClfDecoder_removerBatchEffectDecoder": SuperviseVanillaVAE_regressionClfDecoder_removerBatchEffectDecoder,
    "SuperviseVanillaVAE_removerBatchEffectDecoder": SuperviseVanillaVAE_removerBatchEffectDecoder,
    "SuperviseVanillaVAE_regressionClfDecoder_of_subLatentSpace": SuperviseVanillaVAE_regressionClfDecoder_of_subLatentSpace,
    "SuperviseVanillaVAE_regressionClfDecoder": SuperviseVanillaVAE_regressionClfDecoder,
    "SuperviseVanillaVAE_noCLFDecoder": SuperviseVanillaVAE_noCLFDecoder,
    'SuperviseVanillaVAE': SuperviseVanillaVAE,
    'HVAE': HVAE,
    'LVAE': LVAE,
    'IWAE': IWAE,
    'SWAE': SWAE,
    'MIWAE': MIWAE,
    'VQVAE': VQVAE,
    'DFCVAE': DFCVAE,
    'DIPVAE': DIPVAE,
    'BetaVAE': BetaVAE,
    'InfoVAE': InfoVAE,
    'WAE_MMD': WAE_MMD,
    'VampVAE': VampVAE,
    'GammaVAE': GammaVAE,
    'MSSIMVAE': MSSIMVAE,
    'JointVAE': JointVAE,
    'BetaTCVAE': BetaTCVAE,
    'FactorVAE': FactorVAE,
    'LogCoshVAE': LogCoshVAE,
    'VanillaVAE': VanillaVAE,
    'ConditionalVAE': ConditionalVAE,
    'CategoricalVAE': CategoricalVAE}
