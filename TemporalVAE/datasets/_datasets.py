import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData, OldFormatWarning

from scanpy import settings
from scanpy._utils._doctests import doctest_internet
from scanpy.datasets._utils import check_datasetdir_exists


@doctest_internet
@check_datasetdir_exists
def hEmbryo8_raw(show_datadir=False) -> AnnData:
    r"""Human embryos during peri-implantation.

    This dataset contains the raw read counts by integrating 8 studies. 
    Data available on Zenodo (ref: 15366361, h5ad file_) 

    .. _file: https://zenodo.org/records/15366361/files/rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.h5ad

    .. note::
       This downloads 341 MB of data upon the first call of the function and 
       stores it in :attr:`~TemporalVAE.settings.datasetdir`\ `/hEmbryos8studies.h5ad`.

    Returns
    -------
    Annotated data matrix.

    Examples
    --------
    >>> import TemporalVAE as tvae
    >>> tvae.datasets.hEmbryos8studies()
    AnnData object with n_obs × n_vars = 25003 × 1692
        obs: 'time', 'day', 'dataset_label', 'donor', 'cell_type', 'title', 
            'species', 'n_genes'
        var: 'n_cells'
    
    """
    url = "https://zenodo.org/records/15366361/files/rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.h5ad"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=OldFormatWarning, module="anndata"
        )
        adata = sc.read(
            settings.datasetdir / "rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.h5ad", 
            backup_url=url
        )
    if show_datadir:
        print(settings.datasetdir / 
              "rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.h5ad")
    return adata


@doctest_internet
@check_datasetdir_exists
def hEmbryo8(show_datadir=False) -> AnnData:
    r"""Human embryos during peri-implantation.
    
    This dataset contains the processed read counts by integrating 8 studies, 
    and the inference outputs from TemporalVAE.
    Data available on Zenodo (ref: 15366361, h5ad file_) 

    .. _file: https://zenodo.org/records/15366361/files/rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.filtered.h5ad

    .. note::
       This downloads 448 MB of data upon the first call of the function and 
       stores it in :attr:
       `~TemporalVAE.settings.datasetdir`\ `/hEmbryos8studies_TVAEpost.h5ad`.

    Returns
    -------
    Annotated data matrix.

    Examples
    --------
    >>> import TemporalVAE as tvae
    >>> tvae.datasets.hEmbryos8studies_TVAEpost()
    AnnData object with n_obs × n_vars = 25003 × 1692
        obs: 'time', 'day', 'dataset_label', 'donor', 'cell_type', 'title', 
            'species', 'n_genes', 'tvae_predicted_time'
        var: 'n_cells'
        obsm: 'X_umap', 'tvae_emb'
        layers: 'processed'
    
    """
    url = "https://zenodo.org/records/15366361/files/rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.filtered.h5ad"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=OldFormatWarning, module="anndata"
        )
        adata = sc.read(
            settings.datasetdir / "rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.filtered.h5ad", 
            backup_url=url
        )
    if show_datadir:
        print(settings.datasetdir / 
              "rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.filtered.h5ad")
    return adata
