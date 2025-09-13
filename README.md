# TemporalVAE
Temporal mapping of single cells from time-series atlas with time-predicting VAE

## Reproducibility
For reproducibility of the manuscript's analyses, the scripts for generating 
figures are available at 
[TemporalVAE-reproducibility](
   https://github.com/StatBiomed/TemporalVAE-reproducibility) 
folder/submodule.


## Installation

Quick install can be achieved via pip (python >=3.8; 3.10 to 3.12 were tested)

**Step 0**: create a conda environment and activate it:

```bash
conda create -n tvae python=3.12
conda activate tvae

# Optional: add jupyter lab kernal
pip install ipykernel
python -m ipykernel install --user --name tvae --display-name "tvae"
```

**Step 1**: install TemporalVAE from GitHub:
```bash
# for published version
pip install -U TemporalVAE

# or developing version
pip install -U git+https://github.com/StatBiomed/TemporalVAE
```

## Quick Usage

Reference examples can be found at [examples](./examples) folder, including

* training: [hEmbryo8_training.ipynb](./examples/hEmbryo8_training.ipynb)

* cross-validation: [hEmbryo8_Xiang_CV.ipynb](./examples/hEmbryo8_Xiang_CV.ipynb)

* predicting: [TO BE ADDED]

## Future plan for easier use

Here are the future plan for easier use (TO IMPLEMENT):

1. Import TemporalVAE and create an object of the class TVAE.

```python
import TemporalVAE as tvae

tvae_model = tvae.TVAE()
tvae_model.fit(X_atlas, t_atlas)

# predict query or training data
Z_query, y_query = tvae_model.predict(X_query)
Z_atlas, y_atlas = tvae_model.predict(X_atlas)
```

2. Map to the same 
   [UMAP](https://umap-learn.readthedocs.io/en/latest/api.html#umap.umap_.UMAP) 
   as the reference data

```python
import UMAP

umap_model = UMAP.umap()
umap_model.fit(Z_atlas)

atlas_umap = umap_model.transform(Z_atlas)
query_umap = umap_model.transform(Z_query)
```


## Reference

> Liu Y., Cai F., Barile M., Chang Y., Cao D., and Huang Y. "TemporalVAE: 
  atlas-assisted temporal mapping of time-series single-cell transcriptomes 
  during embryogenesis." Nature Cell Biology, 2025 (in press).
