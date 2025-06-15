from .preprocessing import Preprocessing, transform_labels
from .model import BatchSGDModel, PsupertimeBaseModel
from .parameter_search import RegularizationSearchCV
from .plots import (plot_grid_search,
                    plot_identified_gene_coefficients,
                    plot_labels_over_psupertime,
                    plot_model_perf,
                    plot_identified_genes_over_psupertime)

import datetime
import sys
import warnings
from typing import Iterable, Union

import numpy as np
from sklearn import metrics
import anndata as ad
from scanpy import read_h5ad


class Psupertime:

    def __init__(self,
                 method="proportional",
                 max_memory=None,
                 n_folds=5,
                 n_jobs=5,
                 n_batches=1,
                 verbosity=1,
                 regularization_params=dict(),
                 preprocessing_params=dict(),
                 estimator_class=BatchSGDModel,
                 estimator_params=dict()):

        self.verbosity = verbosity

        # statistical method
        method_options = {"proportional", "forward", "backward"}
        self.method = method
        if self.method not in method_options:
            raise ValueError("Parameter method must be one of %s. Received: %s." % (method_options, self.method))

        # grid search params
        self.n_jobs = n_jobs
        self.n_folds = n_folds

        # model params
        self.max_memory = max_memory
        if self.max_memory is not None:
            warnings.warn("Parameter `max_memory` is currently not implemented. Try setting n_batches directly to control the memory usage.")
        self.n_batches = n_batches

        # TODO: Implement preprocessing as Pipeline with GridSearch -> requires fit functions to take anndata.AnnData
        if not isinstance(preprocessing_params, dict):
            raise ValueError("Parameter estimator_params is not of type dict. Received: ", preprocessing_params)

        self.preprocessing = Preprocessing(**preprocessing_params)

        # Validate estimator params and instantiate model
        if not isinstance(estimator_params, dict):
            raise ValueError("Parameter estimator_params is not of type dict. Received: %s" % estimator_params)

        self.estimator_params = estimator_params
        self.estimator_params["n_batches"] = self.n_batches
        self.estimator_params["method"] = self.method
        self.model = None  # not fitted yet

        if not issubclass(estimator_class, PsupertimeBaseModel):
            raise ValueError("Parameter estimator_class does not inherit PsupertimeBaseModel. Received: ", estimator_class)

        if not isinstance(regularization_params, dict):
            raise ValueError("Parameter estimator_params is not of type dict. Received: ", regularization_params)

        regularization_params["n_jobs"] = regularization_params.get("n_jobs", self.n_jobs)
        regularization_params["n_folds"] = regularization_params.get("n_folds", self.n_folds)
        regularization_params["estimator"] = estimator_class
        self.grid_search = RegularizationSearchCV(**regularization_params)

    def check_is_fitted(self, raise_error=False):
        is_fitted = isinstance(self.model, PsupertimeBaseModel) and self.model.is_fitted_

        if raise_error and not is_fitted:
            ValueError("Invalid estimator class or model not fitted yet. Did you call run() already?")
        else:
            return is_fitted

    def run(self, adata: Union[ad.AnnData, str], ordinal_data: Union[Iterable, str]):

        start_time = datetime.datetime.now()

        # TODO: respect verbosity setting everywhere

        # Validate adata or load the filename
        if isinstance(adata, str):
            filename = adata
            adata = read_h5ad(filename)

        elif not isinstance(adata, ad.AnnData):
            raise ValueError("Parameter adata must be a filename or anndata.AnnData object. Received: ", adata)

        print("Input Data: n_genes=%s, n_cells=%s" % (adata.n_vars, adata.n_obs))

        # Validate the ordinal data
        if isinstance(ordinal_data, str):
            column_name = ordinal_data
            if column_name not in adata.obs.columns:
                raise ValueError("Parameter ordinal_data is not a valid column in adata.obs. Received: ", ordinal_data)

            ordinal_data = adata.obs.get(column_name)

        elif isinstance(ordinal_data, Iterable):
            if len(ordinal_data) != adata.n_obs:
                raise ValueError("Parameter ordinal_data has invalid length. Expected: %s Received: %s" % (len(ordinal_data), len(adata.n_obs)))

        adata.obs["ordinal_label"] = transform_labels(ordinal_data)

        # Run Preprocessing
        print("Preprocessing", end="\r")
        adata = self.preprocessing.fit_transform(adata)
        print("Preprocessing: done. mode='%s', n_genes=%s, n_cells=%s" % (self.preprocessing.select_genes, adata.n_vars, adata.n_obs))

        # TODO: Test / Train split required? -> produce two index arrays, to avoid copying the data?

        # Run Grid Search
        print("Grid Search CV: CPUs=%s, n_folds=%s" % (self.grid_search.n_jobs, self.grid_search.n_folds))
        self.grid_search.fit(adata.X, adata.obs.ordinal_label, estimator_params=self.estimator_params)

        # Refit Model on _all_ data
        print("Refit on all data", end="\r")
        self.model = self.grid_search.get_optimal_model("1se")
        self.model.fit(adata.X, adata.obs.ordinal_label)
        acc = metrics.accuracy_score(self.model.predict(adata.X), adata.obs.ordinal_label)
        dof = np.count_nonzero(self.model.coef_)
        print("Refit on all data: done. accuracy=%f.02, n_genes=%s" % (acc, dof))

        # Annotate the data
        self.model.predict_psuper(adata, inplace=True)

        self.is_fitted_ = True
        print("Total elapsed time: ", str(datetime.datetime.now() - start_time))

        return adata

    def refit_and_predict(self, adata, *args, **kwargs):
        self.check_is_fitted(raise_error=True)
        print("Input Data: n_genes=%s, n_cells=%s" % (adata.n_vars, adata.n_obs))
        print("Refit on all data", end="\r")
        self.model = self.grid_search.get_optimal_model(*args, **kwargs)
        self.model.fit(adata.X, adata.obs.ordinal_label)
        acc = metrics.accuracy_score(self.model.predict(adata.X), adata.obs.ordinal_label)
        dof = np.count_nonzero(self.model.coef_)
        print("Refit on all data: done. accuracy=%f.02, n_genes=%s" % (acc, dof))

        self.model.predict_psuper(adata, inplace=True)

    def predict_psuper(self, *args, **kwargs):
        self.check_is_fitted(raise_error=True)
        return self.model.predict_psuper(*args, **kwargs)

    def plot_grid_search(self, *args, **kwargs):
        self.check_is_fitted(raise_error=True)
        return plot_grid_search(self.grid_search, *args, **kwargs)

    def plot_model_perf(self, *args, **kwargs):
        self.check_is_fitted(raise_error=True)
        return plot_model_perf(self.model, *args, **kwargs)

    def plot_identified_gene_coefficients(self, *args, **kwargs):
        self.check_is_fitted(raise_error=True)
        return plot_identified_gene_coefficients(self.model, *args, **kwargs)

    def plot_identified_genes_over_psupertime(self, *args, **kwargs):
        raise NotImplementedError()
        self.check_is_fitted(raise_error=True)
        return plot_identified_genes_over_psupertime(*args, **kwargs)

    def plot_labels_over_psupertime(self, *args, **kwargs):
        self.check_is_fitted(raise_error=True)
        return plot_labels_over_psupertime(self.model, *args, **kwargs)
