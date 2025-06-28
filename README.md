# TemporalVAE: atlas-assisted temporal mapping of time-series single-cell transcriptomes during embryogenesis

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](https://opensource.org/license/bsd-3-clause)

Contact: Yuanhua Huang, Yijun Liu

Email:  yuanhua@hku.hk


A user-oriented repo is at https://github.com/StatBiomed/TemporalVAE-release with more features to be added.

## Introduction

TemporalVAE is a deep generative model in a dual-objective setting to infer the biological time of cells from a compressed latent space.
We demonstrated its scalability to millions of cells in the mouse development atlas and its high accuracy in atlas-based cell staging on mouse organogenesis across platforms and
during human peri-implantation between in vivo and in vitro conditions.
Furthermore, we showed that our atlas-based time predictor can effectively support RNA velocity modeling over short-time cell differentiation, including hematopoiesis and neuronal
development.

[//]: # (A preprint describing TemporalVAE's algorithms and results is at [bioRxiv]&#40;https://;.)

![](fig1_model_structure.png)

---

## Contents

- [Latest Updates](#latest-updates)
- [Installations](#installation)
- [Reproduce the result in manuscript](#Reproduce the result in manuscript)

## Latest Updates

* v0.1 (May, 2024): Initial release.
* v0.2 (May, 2024)

---

## Installation

To install TemporalVAE, python 3.10.9 is required and follow the instruction

1. Install <a href="https://docs.conda.io/projects/miniconda/en/latest/" target="_blank">Miniconda3</a> if not already available.
2. Clone this repository:

```bash
  git clone https://github.com/StatBiomed/TemporalVAE
```

3. Navigate to `TemporalVAE` directory:

```bash
  cd TemporalVAE
```

4. (5-10 minutes) Create a conda environment (`TemporalVAE-V1.0`) with the required dependencies with two environment configuration files.  `env_necessary.yml` inclueds minimal essential dependencies and `env_all.yml` includes complete development environment.

```bash
# For minimal production environment:
conda env create -f env_necessary.yml
```
If you encounter any pcks version issues, please check `env_all.yml` for more version information.

5. Activate the `TemporalVAE` environment you just created:

```bash
  conda activate TemporalVAE-V1.0
```

[//]: # (6. Install **pytorch**: You may refer to [pytorch installtion]&#40;https://pytorch.org/get-started/locally/&#41; as needed. For example, the command of installing a **cpu-only** pytorch)

[//]: # (   is:)

[//]: # ()
[//]: # (```bash)

[//]: # (conda install pytorch torchvision torchaudio cpuonly -c pytorch)

[//]: # (```)

---

## Reproduce the result in manuscript

The code is in folder named by figure-index.

### Figure 2:
Compare the TemporalVAE with baseline methods on three small datasets cited in [Psupertime](https://academic.oup.com/bioinformatics/article/38/Supplement_1/i290/6617492) mansucript.
1. Preprocess three datasets by the code described in [preprocess_data_fromPsupertimeManuscript.md](demo/Fig2_TemproalVAE_against_benchmark_methods/preprocess_data_fromPsupertimeManuscript.md).
2. run the code of each benchmarking method, then run [plotFig2_check_corr.py](demo/Fig2_TemproalVAE_against_benchmark_methods/plotFig2_check_corr.py) to generate Fig2.
### Figure 3:
1. Preprocess the mouse atlas data and mouse stereo data by

```bash
python -u Fig3_mouse_data/preprocess_data_mouse_embryonic_development_combineData.py
python -u Fig3_mouse_data/preprocess_data_mouse_embryo_stereo.py
```

2. Reproduce the result of **Figure3.A&B** and save results in folder _results/230827_trainOn_mouse_embryonic_development_kFold_testOnYZdata0809_

```bash
python -u Fig3_mouse_data/TemporalVAE_kFoldOn_mouseAtlas.py
--result_save_path=230827_trainOn_mouse_embryonic_development_kFold_testOnYZdata0809
--vae_param_file=supervise_vae_regressionclfdecoder_mouse_stereo
--file_path=/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000
--time_standard_type=embryoneg5to5
--train_epoch_num=100  --kfold_test --train_whole_model
> logs/log.log
```

2. Plot **Figure3.A&B** with the result in _results/230827_trainOn_mouse_embryonic_development_kFold_testOnYZdata0809_, please check _Fig3_mouse_data/plot_figure3AB.ipynb_

3. **Figure3.C**: Compare TemporalVAE with LR, PCA, RF on mouse atlas data, please check _Fig3_mouse_data/LR_PCA_RF_kFoldOn_mouseAtlas.ipynb_

4. **Figure3.D&E**: Models train on mouse atlas data and predict on mouse stereo-seq data, please check _Fig3_mouse_data/TemporalVAE_LR_PCA_RF_directlyPredictOn_mouseStereo.ipynb_
   or run code _Fig3_mouse_data/TemporalVAE_LR_PCA_RF_directlyPredictOn_mouseStereo.py_ on console.

### Figure 4:

1. Download original data of eight published human datasets (See details in Supplementary file). Integrate the raw dataset by

```bash
python -u Fig4_human_data/integration_humanEmbryo_Z_C_Xiao_M_P_Liu_Tyser_Xiang.py
```

2. **Figure 4.A-c**: Performance of TemporalVAE by training on six training datasets and test on two hold-out test dataset by

```bash
python -u Fig4_human_data/TemporalVAE_humanEmbryo_ref6Dataset_queryOnXiang_Tyser.py
```

2. **Sfig**: K-fold test on xiang19 dataset by:
```bash
python -u Fig4_human_data/TemporalVAE_humanEmbryo_kFoldOn_xiang19.py
```
### Figure 5:
1. Preprocess Marmoset and Cynomolgus data by
```bash
python -u Fig5_crossSpecies/preprocess_data_marmoset_inVivo.py
python -u Fig5_crossSpecies/preprocess_data_Cyno.py
```
2. **Figure5.A-D**: Performance of TemporalVAE on cross species prediction by
```bash
python -u Fig5_crossSpecies/TemporalVAE_crossSpecies_referenceMelania_queryOnCynoAndMarmoset.py
```
### Figure 6:
Identification of temporally sensitive genes by in silico perturbation.Here, we focus on the mouse embryo atlas as a showcase, thanks to its data consistency and broader time range.
```bash
python -u Fig6_identify_keyGenes/TemporalVAE_identify_keyGenes_mouseAtlas.py
python -u Fig6_identify_keyGenes/plot_perturbution_results.py
```

## Todo

### Figure 5 - RNA velocity:
1. The data is from paper <Systematic reconstruction of the cellular trajectories of mammalian embryogenesis.>.
2. 1 **Figure 5. C&E** is the data of hematopoiesis cells, please check _Fig5_RNA_velocity/VAE_mouse_fineTune_Train_on_U_pairs_S_hematopoiesis.ipynb_ or run code on console:
```bash
python -u Fig5_RNA_velocity/TemporalVAE_mouse_fineTune_Train_on_U_pairs_S.py --sc_file_name=240108mouse_embryogenesis/hematopoiesis --clf_weight=0.2
```
2. 2 **Figure 5. D&F** is the data of neuron cells, please check _Fig5_RNA_velocity/VAE_mouse_fineTune_Train_on_U_pairs_S_neuron.ipynb_ or run code on console:
```bash
python -u Fig5_RNA_velocity/TemporalVAE_mouse_fineTune_Train_on_U_pairs_S.py --sc_file_name=240108mouse_embryogenesis/neuron --clf_weight=0.1
```
3. The scVelo result in **Figure 5. E&F** is base on the _.ipynb_ code provided by the dataset's paper, please check _Fig5_RNA_velocity/scVelo_hematopoiesis.ipynb_ and _Fig5_RNA_velocity/scVelo_neuron.ipynb_
[//]: # (Build a well-structured software packages)










