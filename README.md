# TemporalVAE
Temporal mapping of single cells from time-series atlas

## Reproducibility
For reproducibility of the manuscript's analyses, the scripts for generating figures are available at 
[TemporalVAE-reproducibility](https://github.com/StatBiomed/TemporalVAE-reproducibility) 
folder/submodule.


## Installation

Quick install can be achieved via pip (python >=3.8)

```bash
# for published version
pip install -U TemporalVAE

# or developing version
pip install -U git+https://github.com/StatBiomed/TemporalVAE
```

## Quick Usage

More examples can be found at [examples](./examples) folder, including
[TemporalVAE_demo.ipynb](./examples/TemporalVAE_demo.ipynb)


Here is a quick start (TO IMPLEMENT):

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

> Y. Liu et al. Atlas-assisted temporal mapping of time-series single-cell transcriptomes during embryogenesis. (to appear)
