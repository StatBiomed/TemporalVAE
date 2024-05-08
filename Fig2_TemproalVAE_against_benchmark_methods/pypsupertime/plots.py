from .parameter_search import RegularizationSearchCV
from .model import PsupertimeBaseModel
from .preprocessing import transform_labels, calculate_weights
import numpy as np
from sklearn import metrics
import seaborn as sns
import anndata as ad
import pandas as pd
from matplotlib import pyplot as plt
import warnings
from typing import Optional, Tuple, Iterable


def plot_grid_search(grid_search: RegularizationSearchCV, title: str="Grid Search Results", figsize: Tuple[int, int]=(16,4), y_log_weights: bool=False):
    """Plot the weights, scores, and degrees of freedom for a parameter search as a function of the 
    regularization path. Indicates the best model and most sparse model within one standard error of the 
    best score based on the averaged cross validation scores. 

    :param grid_search: grid search object
    :type grid_search: RegularizationSearchCV
    :param title: Figure title, defaults to "Grid Search Results"
    :type title: str, optional
    :param figsize: Size of the matplotlib figure, defaults to (16,4)
    :type figsize: tuple, optional
    :param y_log_weights: log scale y axis of weights plot, defaults to False
    :type y_log_weights: boolean
    :return: figure instance 
    :rtype: matplotlib.pyplot.Figure
    """
    if not isinstance(grid_search, RegularizationSearchCV):
        raise ValueError("The first argument needs to be a completed GridSearch")
    
    if not grid_search.is_fitted_:
        raise ValueError("The grid_search must be run first! Did you call the `.fit()` method?")
    
    best_lambda, best_idx = grid_search.get_optimal_regularization("best")
    print("Best idx:", best_idx, "Best Score:", grid_search.scores[best_idx], "Best Lambda:", grid_search.reg_path[best_idx], "Scores std:", np.std(grid_search.scores))
    ose_lambda, ose_idx = grid_search.get_optimal_regularization("1se")
    print("1SE idx:", ose_idx, "1SE Score:", grid_search.scores[ose_idx], "1SE Lambda:", grid_search.reg_path[ose_idx])

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(131)
    fitted_weights = np.array([np.array(e.coef_).flatten() for e in grid_search.fitted_estimators])
    ax1.plot(fitted_weights)
    ax1.axvline(best_idx, ymin=0, ymax=1, color="red", label="best", ls="--")
    ax1.axvline(ose_idx, ymin=0, ymax=1, color="blue", label="1se", ls="--")
    ax1.set_xlabel("regularization iteration")
    ax1.set_ylabel("weights")
    if y_log_weights: ax1.set_yscale("log")
    ax1.legend()

    ax3 = fig.add_subplot(132)
    ax3.errorbar(x=np.arange(len(grid_search.train_scores))+0.1, y=grid_search.train_scores, yerr=grid_search.train_scores_std, elinewidth=0.5, color="tab:blue", ecolor="skyblue", barsabove=True, label="train scores")
    ax3.errorbar(x=np.arange(len(grid_search.scores)), y=grid_search.scores, yerr=grid_search.scores_std, elinewidth=0.5, color="tab:orange", ecolor="moccasin", barsabove=True, label="test scores")
    ax3.axvline(best_idx, ymin=0, ymax=1, color="red", label="best", ls="--")
    ax3.axvline(ose_idx, ymin=0, ymax=1, color="blue", label="1se", ls="--")
    ax3.set_xlabel("regularization iteration")
    ax3.set_ylabel("mean cv score")
    ax3.legend()

    ax2 = fig.add_subplot(133)
    ax2.plot(grid_search.dof)
    ax2.axvline(best_idx, ymin=0, ymax=1, color="red", label="best", ls="--")
    ax2.axvline(ose_idx, ymin=0, ymax=1, color="blue", label="1se", ls="--")
    ax2.set_xlabel("regularization iteration")
    ax2.set_ylabel("Degrees of Freedom")

    fig.suptitle(title)

    return fig


def plot_model_perf(model: PsupertimeBaseModel, train: Tuple[Iterable, Iterable], test: Optional[Tuple[Iterable, Iterable]]=None, title: str="Model Predictions", figsize: Tuple[int, int]=(10, 4)):
    """Print model performance statistics and plot the confusion matrices for the 
    test and training prediction, respectively.

    :param model: Fitted instance of a PsupertimeBaseModel
    :type model: PsupertimeBaseModel
    :param train: X_train, y_train
    :type train: tuple
    :param test: X_test, y_test
    :type test: tuple, optional
    :param title: Figure title, defaults to "Model Predictions"
    :type title: str, optional
    :param figsize: Size of the matplotlib figure, defaults to (16,4)
    :type figsize: tuple, optional
    :return: figure instance
    :rtype: matplotlib.pyplot.Figure
    """
    if not isinstance(model, PsupertimeBaseModel):
        raise ValueError("The first argument needs to be a fitted sklearn model")
    
    if not model.is_fitted_:
        raise ValueError("The grid_search must be run first! Did you call the `.fit()` method?")
    
    if not isinstance(train, tuple) or (test is not None and not isinstance(test, tuple)):
        raise ValueError("The parameters `train`, `test` are expected to be tuples of np.array")
    
    print("Model Degrees of freedom", np.count_nonzero(np.array(model.coef_).flatten()))

    if test is not None:

        labels = np.unique(np.concatenate([y_test, y_train]))
        
        X_train, y_train = train
        X_test, y_test = test
        
        y_train_trans = transform_labels(y_train)
        weights_train = calculate_weights(y_train_trans)

        y_test_trans = transform_labels(y_test)
        weights_test = calculate_weights(y_test_trans)

        print("Train:")
        print("Accuracy:", metrics.accuracy_score(y_train_trans, model.predict(X_train)))
        print("Balanced accuracy:", metrics.balanced_accuracy_score(y_train_trans, model.predict(X_train)))
        print("Mean absolute delta:", metrics.mean_absolute_error(y_train_trans, model.predict(X_train), sample_weight=weights_train))
        print("Test:")
        print("Accuracy:", metrics.accuracy_score(y_test_trans, model.predict(X_test)))
        print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test_trans, model.predict(X_test)))
        print("Mean absolute delta:", metrics.mean_absolute_error(y_test_trans, model.predict(X_test), sample_weight=weights_test))
            
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        sns.heatmap(metrics.confusion_matrix(y_test_trans, model.predict(X_test)), annot=True, ax=ax1)
        sns.heatmap(metrics.confusion_matrix(y_train_trans, model.predict(X_train)), annot=True, ax=ax2)
        ax1.set_xlabel("Test-Split Preditions")
        ax1.set_ylabel("Ground Truth Labels")
        ax2.set_xlabel("Train-Split Predictions")

    else:
        X, y = train
        labels = np.unique(y)
        y_trans = transform_labels(y)
        weights = calculate_weights(y_trans)

        print("Accuracy:", metrics.accuracy_score(y_trans, model.predict(X)))
        print("Balanced accuracy:", metrics.balanced_accuracy_score(y_trans, model.predict(X)))
        print("Mean absolute delta:", metrics.mean_absolute_error(y_trans, model.predict(X), sample_weight=weights))
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        sns.heatmap(metrics.confusion_matrix(y_trans, model.predict(X)), annot=True, ax=ax)
        ax.set_ylabel("Ground Truth Labels")
        ax.set_xlabel("Predictions")


    fig.suptitle(title)

    return fig


def plot_identified_gene_coefficients(model: PsupertimeBaseModel, anndata: ad.AnnData,  n_top: int=30, figsize: Tuple[int, int]=(6,6), *args, **kwargs):
    """Extracts gene weigts from model and plots the `n_top` highest gene coefficient sorted by their abs value
    as horizontal bars.

    :param model: fitted model
    :type model: PsupertimeBaseModel
    :param anndata: annotation data
    :type anndata: ad.AnnData
    :param n_top: number of highest coefficients to plot, defaults to 30
    :type n_top: int, optional
    :param figsize: Size of the matplotlib figure, defaults to (6,6)
    :type figsize: tuple, optional
    :return: Returns the plot as matplotlib figure
    :rtype: matplotlib.pyplot.Figure
    """
    if not isinstance(anndata, ad.AnnData):
        raise ValueError("anndata must be an instanec of anndat.AnnData")

    if not isinstance(model, PsupertimeBaseModel):
        raise ValueError("model must be an instance of PsupertimeBaseModel")

    # always recalculate gene_weights
    psuper_weights_key = "psupertime_weight"
    var_copy = model.gene_weights(anndata, inplace=False)

    sorted_idx = np.argsort(np.abs(var_copy[psuper_weights_key]))
    max_val = np.abs(var_copy[psuper_weights_key][sorted_idx][-1])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    var_copy[psuper_weights_key][sorted_idx[-n_top:]].plot.barh(ax=ax, width=0.25, color="black")
    ax.bar_label(ax.containers[0])
    ax.set_xlim((-(max_val * 1.5), max_val * 1.5))
    ax.axvline(0, color="black", ls="--")

    return fig


def plot_identified_genes_over_psupertime(n = 20, *args, **kwargs):
    raise NotImplementedError()


def plot_labels_over_psupertime(model: PsupertimeBaseModel, anndata: ad.AnnData, label_key: str, figsize: Tuple[int, int]=(10, 5), *args, **kwargs) -> plt.Figure:
    """Distirbution of cells, grouped by their ordinal label as a function of the predicted pseudotime.

    :param model: fitted psupertime model
    :type model: PsupertimeBaseModel
    :param anndata: annotation data instance
    :type anndata: ad.AnnData
    :param label_key: label of the column with ordinal labels in anndata.obs
    :type label_key: str
    :param figsize: Size of matplotlib figure, defaults to (10, 5)
    :type figsize: tuple, optional
    :return: Matplotlib figure
    :rtype: matplotlib.pyplot.Figure
    """

    if not isinstance(anndata, ad.AnnData):
        raise ValueError("anndata must be an instanec of anndat.AnnData")
    
    if not hasattr(anndata.obs, label_key):
        raise ValueError("anndata.obs does not contain column with key '%s'" % label_key)

    if not isinstance(model, PsupertimeBaseModel):
        raise ValueError("model must be an instance of PsupertimeBaseModel")

    # always recalculate psupertime
    psupertime_key = "psupertime"
    obs_copy = model.predict_psuper(anndata, inplace=False)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    palette='RdBu'
    col_vals = sns.color_palette(palette, len(obs_copy[label_key].unique()))
    sns.kdeplot(data=obs_copy, x=psupertime_key, fill=label_key, hue=label_key, alpha=0.5,
                    palette=col_vals, legend=True, ax=ax)

    thresholds = model.intercept_ * -1  # inverted thetas
    for x, c in zip(thresholds, col_vals):
        ax.axvline(x=x, color=c)

    ax.set_xlabel("Psupertime")
    ax.set_ylabel("Density")
    sns.despine()

    return fig
