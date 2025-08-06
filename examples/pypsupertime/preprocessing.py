import warnings
import numpy as np
from typing import Union, Iterable, Optional
import scanpy as sc
import anndata as ad
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import class_weight
from scipy import sparse

numeric = Union[int, float, np.number]


def restructure_y_to_bin(y_orig):
    """ 
    The given labels are converted to a binary representation,
    such that the threshold from 0-1 corresponds from changing from label
    $l_i$ to $l_{i+1}$. 
    $k$ copies of the label vector are concatenated such that for every
    vector $j$ the labels  $l_i$ with $i<j$ are converted to 0 and the 
    labels $i < j$ are converted to 1.

    :param y_orig: Original data set labels
    :type y_orig: Iterable
    :return: Restructured Labels: array of length n * k
    :rtype: numpy array
    """
    y_classes = np.unique(y_orig)
    k = len(y_classes)

    y_bin = []
    for ki in range(1, k):
        thresh = y_classes[ki]
        y_bin += [int(x >= thresh) for x in y_orig]

    y_bin = np.array(y_bin)

    return y_bin


def restructure_X_to_bin(X_orig, n_thresholds):
    """
    The count matrix is extended with copies of itself, to fit the converted label
    vector. For big problems, it could suffice to have just one label
    vector and perform and iterative training.
    To train the thresholds, $k$ columns are added to the count matrix and
    initialized to zero. Each column column represents the threshold for a
    label $l_i$ and is set to 1, exactly  where that label $l_1$ occurs.

    :param X_orig: input data
    :type X_orig: numpy array or sparse.csr_matrix with shape (n_cells, n_genes)
    :param n_thresholds: number of thresholds to be learned - equal to num_unique_labels - 1
    :type n_thresholds: integer
    :return: Restructured matrix of shape (n_cells * n_thresholds, n_genes + n_thresholds)
    :rtype: numpy array
    """

    n = X_orig.shape[0]
    binarized_index = np.arange(n * n_thresholds)
    index_mod_n = binarized_index % n
    thresholds = np.identity(n_thresholds)

    if sparse.issparse(X_orig):
        thresholds = sparse.csr_matrix(thresholds)
        X_bin = sparse.hstack((X_orig[index_mod_n], thresholds[binarized_index // n]))
    else:
        X_bin = np.hstack((X_orig[index_mod_n], thresholds[binarized_index // n]))

    return X_bin


def transform_labels(y: Iterable[numeric]):
    """
    Transforms a target vector, such that it contains successive labels starting at 0.

    :param y: Iterable containing the ordinal labels of a dataset. Note: Must be number (int, float, np.number)!
    :type y: Iterable[number]
    :return: Numpy array with the labels converted
    :rtype: numpy.array
    """

    # convert to numeric
    try:
        y = np.array(y).astype("float32")
    except ValueError as e:
        print(e)
        raise ValueError("Error Converting labels to numeric values")

    labels = np.unique(y)
    ordering = labels.argsort()
    y_trans = np.zeros_like(y)
    for i, el in enumerate(y):
        for l, o in zip(labels, ordering):
            if el == l:
                y_trans[i] = o

    return y_trans


def calculate_weights(y):
    """
    Calculates weights from the classes in y.
    Returns an array the same length as y,
    where the class is replaced with their respective weight

    Calculates balanced class weights according to
    `n_samples / (n_classes * np.bincount(y))`
    as is done in sklearn.
    """
    classes = np.unique(y)
    weights = class_weight.compute_class_weight("balanced", classes=classes, y=y)

    transf = dict(zip(classes, weights))

    return np.array([transf[e] for e in y])


def smooth(adata, knn=10, inplace=True):
    """
    Smoothes data by averaging over the k nearest neighbors as a denoising step.

    :param adata: annotation data object with data in `adata.X`
    :type adata: anndata.AnnData
    :param knn: number of nearest neighbors to smooth over, defaults to 10
    :type knn: int, optional
    :param inplace: Perform the smoothing on the original anndata object, overwriting the existing `adata.X`.
      If set to False, a copy of `adata` is created, defaults to True
    :type inplace: bool, optional
    :return: annotation data instance
    :rtype: anndata.AnnData
    """

    if sparse.issparse(adata.X):
        warnings.warn(
            "Smoothing requires conversion of input data to dense format. If n_obs is large, this can be very memory intensive!")
        adata.X = np.array(adata.X.todense())

    # corellate all cells
    cor_mat = np.corrcoef(adata.X)

    # calculate the ranks of cell correlations
    order_mat = np.argsort(cor_mat, axis=1)
    del (cor_mat)  # Delete unnecessary copy

    rank_mat = np.argsort(order_mat, axis=1)
    del (order_mat)  # Delete unnecessary copy

    # indicate the knn closest neighbours
    idx_mat = rank_mat <= knn
    del (rank_mat)  # Delete unnecessary copy

    # calculate the neighborhood average
    avg_knn_mat = idx_mat / np.sum(idx_mat, axis=1, keepdims=True)
    assert np.all(np.isclose(np.sum(avg_knn_mat, axis=1), 1))

    imputed_mat = np.dot(avg_knn_mat, adata.X)

    if inplace:
        adata.X = imputed_mat

    else:
        adata = adata.copy()
        adata.X = imputed_mat
        return adata


class Preprocessing(BaseEstimator, TransformerMixin):
    """
    Preprocessing for anndata.AnnData instances that implements an
    [sklearn transformer](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html).

    The following steps are performed in that order:
        1. Log+1 transformation
        2. Filtering of genes by minimum expression
        3. Selection of genes by one of the following flavors:
            - high variability (seurat flavor),
            - role as transcription factors,
            - matching a list of user-selected IDs, or
            - using all genes
        4. Denoising by smoothing over k nearest neighbors
        5. Normalization incorporating all genes
        6. Scaling to unit variance and shiftig to mean zero

    Except for the filtering by minimum expression and selection of genes, steps can be skipped at will and for
    the best outcome it is recommended that the user performs their own preprocessing instead of relying on our recipe.

    The [scanpy](https://scanpy.readthedocs.io/en/stable/) package is used to perform each of the steps above.

    :param log: Perform log transformation after adding 1 where counts are 0, defaults to False
    :type log: bool, optional
    :param scale: Scale to unit variance and shifting to mean zero, defaults to True
    :type scale: bool, optional
    :param normalize: Normalize over all counts, defaults to False
    :type normalize: bool, optional
    :param smooth: Smooth over nearest neighbor, defaults to False
    :type smooth: bool, optional
    :param smooth_knn: Number of neighbors to average over, if `smooth` is set to True. Defaults to 10
    :type smooth_knn: int, optional
    :param select_genes: Method of selecting genes. Must be one of
        - `"all"`: use all genes. This is ssed by default, but is the most computationally expensive
        - `"hvg"`: highly variable genes according to the seurat implementation
        - `"tf_mouse"`: Use list of mouse transcription factors. Not implemented, yet
        - `"tf_human"`: Use list of human transcription factors. Not implemented, yet
        - `"list"`: Use a user-curated list.
    :type select_genes: str, optional
    :param gene_list: Iterable of user-selected genes, only used if `select_genes` equals `"list"`. Defaults to None
    :type gene_list: Optional[Iterable], optional
    :param min_gene_mean: Minimum average counts cutoff per gene, defaults to 0.01
    :type min_gene_mean: float, optional
    :param max_gene_mean: Maximum average counts cutoff per gene, defaults to 3. Note: Currently not used!
    :type max_gene_mean: float, optional
    :param hvg_min_dispersion: Miminum dispersion cutoff, only used if `select_genes` equals `"hvg"`
        and `hvg_n_top_genes` unequals `None`. Defaults to 0.5
    :type hvg_min_dispersion: float, optional
    :param hvg_max_dispersion: Maximum dispersion cutoff, only used if `select_genes` equals `"hvg"`
        and `hvg_n_top_genes` unequals `None`. Defaults to np.inf
    :type hvg_max_dispersion: float, optional
    :param hvg_n_top_genes: Number of genes to select, only used if `select_genes` equals `"hvg"`. Defaults to None
    :type hvg_n_top_genes: Optional[int], optional
    :raises ValueError: _description_
    :raises ValueError: _description_
    """

    def __init__(self,
                 log: bool = False,
                 scale: bool = True,
                 normalize: bool = False,
                 smooth: bool = False,
                 smooth_knn: int = 10,
                 select_genes: str = "all",
                 gene_list: Optional[Iterable] = None,
                 min_gene_mean: float = 0.01,
                 max_gene_mean: float = 3,
                 hvg_min_dispersion: float = 0.5,
                 hvg_max_dispersion: float = np.inf,
                 hvg_n_top_genes: Optional[int] = None):

        self.scale = scale
        self.log = log
        self.normalize = normalize
        self.smooth = smooth
        self.smooth_knn = smooth_knn
        self.min_gene_mean = min_gene_mean
        self.max_gene_mean = max_gene_mean
        self.hvg_min_dispersion = hvg_min_dispersion
        self.hvg_max_dispersion = hvg_max_dispersion
        self.hvg_n_top_genes = hvg_n_top_genes

        # Validate gene selection method
        select_genes_options = {"all", "hvg", "tf_mouse", "tf_human", "list"}
        self.select_genes = select_genes
        if self.select_genes not in select_genes_options:
            raise ValueError("Parameter select_genes must be one of %s." % select_genes_options)

        # Validate gene list
        self.gene_list = gene_list
        if self.gene_list is not None:
            if not isinstance(self.gene_list, Iterable):
                raise ValueError("Parameter gene_list must be an Iterable")

            if self.select_genes != "list":
                warnings.warn(
                    "Parameter select_genes was set to '%s' but gene_list was given and will be used" % self.select_genes)
                self.select_genes = "list"

            self.gene_list = np.array(self.gene_list)

    def fit_transform(self, adata: ad.AnnData, y: Optional[Iterable] = None, **fit_params) -> ad.AnnData:
        """Fits transformer to an instance of `anndata.Anndata` and returns a transformed version of it
         with the user-selected preprocessing applied.

         This transformer strictly works on anndata instances, which store data and labels in a single object.
         The parameters `y` is therfor never used, but kept to comply with the sklearn standard.

        :param adata: Annotation data instance to be transformed
        :type adata: anndata.AnnData
        :param y: Only described to conform with the ever used, defaults to None
        :type y: Optional[Iterable], optional
        :raises NotImplementedError: When `select_genes` is set to `tf_human` or `tf_mouse`
        :return: Transformed AnnData
        :rtype: anndata.AnnData
        """

        # log transform: adata.X = log(adata.X + 1)
        if self.log: sc.pp.log1p(adata, copy=False)

        # filter genes by their minimum mean counts
        cell_thresh = np.ceil(self.min_gene_mean * adata.n_obs)
        sc.pp.filter_genes(adata, min_cells=cell_thresh)

        # select highly-variable genes
        if self.select_genes == "hvg":
            sc.pp.highly_variable_genes(
                adata, flavor='seurat', n_top_genes=self.hvg_n_top_genes,
                min_disp=self.hvg_min_dispersion, max_disp=self.hvg_max_dispersion, inplace=True
            )
            adata = adata[:, adata.var.highly_variable].copy()

        # select mouse transcription factors
        elif self.select_genes == "tf_mouse":
            raise NotImplementedError("select_genes='tf_mouse'")

        # select human transcription factors
        elif self.select_genes == "tf_human":
            raise NotImplementedError("select_genes='tf_human'")

        # select curated genes from list
        # TODO: Check for array
        elif self.select_genes == "list":
            adata = adata[:, self.gene_list].copy()

        # select all genes
        elif self.select_genes == "all":
            pass

        # smoothing over neighbors to denoise data
        if self.smooth:
            smooth(adata, knn=self.smooth_knn, inplace=True)

        # normalize over all counts
        # this helps keep the parameters small
        if self.normalize: sc.pp.normalize_total(adata)

        # scale to unit variance and shift to zero mean
        if self.scale: sc.pp.scale(adata)

        return adata