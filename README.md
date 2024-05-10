
[//]: # (<div align="center">)

[//]: # (    <img src="images/stemVAE_logo.png" width = "350" alt="stemVAE">)

[//]: # (</div>)

# TemporalVAE: atlas-assisted temporal mapping of time-series single-cell transcriptomes during embryogenesis

[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/mit/) 

Contact: Yuanhua Huang, Yijun Liu

Email:  yuanhua@hku.hk

## Introduction
 TemporalVAE is a deep generative model in a dual-objective setting to infer the biological time of cells from a compressed latent space.
We demonstrated its scalability to millions of cells in the mouse development atlas and its high accuracy in atlas-based cell staging on mouse organogenesis across platforms and during human peri-implantation between in vivo and in vitro conditions. 
Furthermore, we showed that our atlas-based time predictor can effectively support RNA velocity modeling over short-time cell differentiation, including hematopoiesis and neuronal development.

A preprint describing TemporalVAE's algorithms and results is at [bioRxiv](https://;.



![](./stemVAE/231019model_structure.png)

---


## Contents

- [Latest Updates](#latest-updates)
- [Installations](#installation)
- [Usage](#usage)
    - [Model training](#model-training)
    - [Performance evaluation](#performance-evaluation)
    - [Spatial inference](#spatial-inference)
   

## Latest Updates
* v0.1 (May, 2025): Initial release.
---
## Installation
To install stemVAE, python 3.9 is required and follow the instruction
1. Install <a href="https://docs.conda.io/projects/miniconda/en/latest/" target="_blank">Miniconda3</a> if not already available.
2. Clone this repository:
```bash
  git clone https://github.com/StatBiomed/TemporalVAE
```
3. Navigate to `TemporalVAE` directory:
```bash
  cd TemporalVAE
```
4. (5-10 minutes) Create a conda environment with the required dependencies:
```bash
  conda env create -f environment.yml
```
5. Activate the `stemVAE` environment you just created:
```bash
  conda activate Temporal
```
6. Install **pytorch**: You may refer to [pytorch installtion](https://pytorch.org/get-started/locally/) as needed. For example, the command of installing a **cpu-only** pytorch is:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
## Reproduce the result in manuscript
The code is in folder named by figure-index
### Figure 2: Compare the TemporalVAE with baseline methods in three small datasets cited in Psupertime mansucript

### Figure 3: 
1. Preprocess the mouse atlas data and mouse stereo data by
```bash
python -u Fig3_mouse_data/preprocess_data_mouse_embryonic_development_combineData.py
python -u Fig3_mouse_data/preprocess_data_mouse_embryo_stereo.py
```
2. 2 Reproduce the result of **Figure3.A&B** and save results in folder _results/230827_trainOn_mouse_embryonic_development_kFold_testOnYZdata0809_
```bash
python -u Fig3_mouse_data/VAE_mouse_kFoldOn_mouseAtlas.py 
--result_save_path=230827_trainOn_mouse_embryonic_development_kFold_testOnYZdata0809
--vae_param_file=supervise_vae_regressionclfdecoder_mouse_stereo
--file_path=/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000 
--time_standard_type=embryoneg5to5
--train_epoch_num=100  --kfold_test --train_whole_model
> logs/log.log
```
2. 2 Plot **Figure3.A&B** with the result in _results/230827_trainOn_mouse_embryonic_development_kFold_testOnYZdata0809_, please check _Fig3_mouse_data/plot_figure3AB.ipynb_

2. 3 **Figure3.C**: Compare TemporalVAE with LR, PCA, RF on mouse atlas data, please check _Fig3_mouse_data/LR_PCA_RF_kFoldOn_mouseAtlas.ipynb_

3. **Figure3.D&E**: Models train on mouse atlas data and predict on mouse stereo-seq data, please check _Fig3_mouse_data/TemporalVAE_LR_PCA_RF_directlyPredictOn_mouseStereo.ipynb_ or run code _Fig3_mouse_data/TemporalVAE_LR_PCA_RF_directlyPredictOn_mouseStereo.py_ on console.


### Figure 4:
1. Preprocess the raw dataset by
```bash
python -u Fig4_human_data/preprocess_humanEmbryo_xiang2019data.py
python -u Fig4_human_data/preprocess_humanEmbryo_PLOS.py
python -u Fig4_human_data/preprocess_humanEmbryo_CS7_Tyser.py
```
2. **Figure 4.A**: K-fold test on xiang19 dataset, please check _Fig4_human_data/vae_humanEmbryo_xiang19.ipynb_ or run code on console:
```bash
python -u Fig4_human_data/vae_humanEmbryo_xiang19.py --file_path=/240322Human_embryo/xiang2019/hvg500/
```
3. **Figure 4.B**: Temporal trained on xiang19 dataset and predict on Lv19 dataset, please check _Fig4_human_data/LR_PCA_RF_directlyPredictOn_humanEmbryo_PLOS.ipynb_ or run code _Fig4_human_data/LR_PCA_RF_directlyPredictOn_humanEmbryo_PLOS.py_ on console.
4. **Figure 4C&D**: train on 4 in vitro dataset and predict on one in vivo dataset, please check _Fig4_human_data/vae_humanEmbryo_Melania.ipynb_ or run code on console:
```bash
python -u Fig4_human_data/vae_humanEmbryo_Melania.py --file_path=/240405_preimplantation_Melania/
```

### Figure 5:

## Usage


StemVAE contains 2 main function: k fold test on dataset; predict on a new donor. And we also provide code to reproduce the result in the paper. 

To check available modules, run:
### prepare the preprocess data:
_Todo list_
### k fold test
The result will save to folder _results_, log file wile save to folder _logs_
```bash
python -u VAE_fromDandan_testOnOneDonor.py 
--vae_param_file=supervise_vae_regressionclfdecoder 
--file_path=preprocess_02_major_Anno0717_Gene0720 --time_standard_type=neg1to1 
--train_epoch_num=100 
--result_save_path=230728_newAnno0717_Gene0720_18donor_2type_plot 
> logs/log.log
```








