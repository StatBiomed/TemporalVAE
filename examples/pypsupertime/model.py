from abc import ABC, ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from scipy import sparse
import anndata as ad
import numpy as np
import pandas as pd
from numpy.random import default_rng
import warnings

from .preprocessing import restructure_X_to_bin, restructure_y_to_bin, transform_labels


# Maximum positive number before numpy 64bit float overflows in np.exp()
MAX_EXP = 709


class PsupertimeBaseModel(ClassifierMixin, BaseEstimator, ABC):
    """
    Abstract Base class to build scikit-learn compatible models for PyPsupertime derived from `sklearn.base.BaseEstimator` and 
    `sklearn.base.ClassifierMixin`.

    Provides methods for restructuring ordinal data into a binary representation and 
    for fitting a nested binary logistic classifier.
    
    Provides predict methods, that uses the fitted binary classifier to estimate the probabilities and labels of
    the ordinal multiclass problem.

    :ivar method: Statistical model used for ordinal logistic regression: One of `"proportional"`, `"forward"` 
     and `"backward"`, corresponding to cumulative proportional odds, forward continuation ratio and
     backward continuation ratio.
    :type method: str 
    :ivar binary_estimator_: sklearn model which is fitted on the ordinal binary logistic classification task
    :type binary_estimator_: sklearn.base.ClassifierMixin
    :ivar regularization: parmeter controlling the sparsity of the model. Wrapper for the respective parameter
     of the nested `binary_estimator_`. Not necessary
    :type regularization: float
    :ivar k_: number of thresholds to be learned, equal to one less than the number of unique ordinal labels 
    :type k_: int

    """
    method: str = "proportional"
    binary_estimator_: BaseEstimator = None
    regularization: float
    random_state: int = 1234
    coef_: np.array
    intercept_: np.array
    k_: int = 0
    classes_: np.array
    is_fitted_: bool = False

    @abstractmethod
    def _binary_estimator_factory(self):
        raise NotImplementedError()
 
    def get_binary_estimator(self):
        """Returns the nested binary model extending `sklearn.base.BaseEstimator`

        :raises ValueError: _description_
        :return: _description_
        :rtype: _type_
        """
        if self.binary_estimator_ is None:
            self.binary_estimator_  = self._binary_estimator_factory()

        if not isinstance(self.binary_estimator_, BaseEstimator):
            raise ValueError("The underlying 'binary_estimator' is not a sklearn.base.BaseEstimator. Got this instead: ", self.binary_estimator_)
        
        return self.binary_estimator_

    def _before_fit(self, data, targets):
        data, targets = check_X_y(data, transform_labels(targets), accept_sparse=True)
        self.classes_ = np.unique(targets)
        self.k_ = len(self.classes_) - 1
        return data, targets
    
    def _after_fit(self, model):
        self.is_fitted_ = True

        # extract the thresholds and weights
        # from the 2D coefficients matrix in the sklearn model
        self.intercept_ = np.array(model.coef_[0, -self.k_:]) + model.intercept_  # thresholds
        self.coef_ = model.coef_[0, :-self.k_]   # weights

    def fit(self, data, targets, sample_weight=None):
        """Template fit function for derived models.

        :param data: 2d data
        :type data: numpy or numpy.sparse matrix
        :param targets: Array-like object with ordinal labels
        :type targets: Iterable
        :param sample_weight: label weights to be used for training and scoring, defaults to None
        :type sample_weight: Iterable, optional
        :return: fitted estimator
        :rtype: PsupertimeBaseModel
        """
        data, targets = self._before_fit(data, targets)

        # convert to binary problem
        data = restructure_X_to_bin(data, self.k_)
        targets = restructure_y_to_bin(targets)

        model = self.get_binary_estimator()
        
        weights = np.tile(sample_weight, self.k_) if sample_weight is not None else None
        model.fit(data, targets, sample_weight=weights)
        self._after_fit(model)

        return self

    def predict_proba(self, X):
        warnings.filterwarnings("once")

        transform = X @ self.coef_        
        logit = np.zeros(X.shape[0] * (self.k_)).reshape(X.shape[0], self.k_)
        
        # calculate logit
        for i in range(self.k_):
            # Clip exponents that are larger than MAX_EXP before np.exp for numerical stability
            # this will cause warnings and nans otherwise!
            temp = self.intercept_[i] + transform
            temp = np.clip(temp, np.min(temp), MAX_EXP)
            exp = np.exp(temp)
            logit[:, i] = exp / (1 + exp)

        prob = np.zeros(X.shape[0] * (self.k_ + 1)).reshape(X.shape[0], self.k_ + 1)
        # calculate differences
        for i in range(self.k_ + 1):
            if i == 0:
                prob[:, i] = 1 - logit[:, i]
            elif i < self.k_:
                prob[:, i] = logit[:, i-1] - logit[:, i]
            elif i == self.k_:
                prob[:, i] = logit[:, i-1]
        
        warnings.filterwarnings("always")
        return prob
    
    def predict(self, X):
        return np.apply_along_axis(np.argmax, 1, self.predict_proba(X))

    def predict_psuper(self, anndata: ad.AnnData, inplace=True):
        
        transform = anndata.X @ self.coef_
        predicted_labels = self.predict(anndata.X)      

        if inplace:
            anndata.obs["psupertime"] = transform
            anndata.obs["predicted_label"] = predicted_labels
        
        else:
            obs_copy = anndata.obs.copy()
            obs_copy["psupertime"] = transform
            obs_copy["predicted_label"] = predicted_labels
            return obs_copy
    
    def gene_weights(self, anndata: ad.AnnData, inplace=True):
        if inplace:
            anndata.var["psupertime_weight"] = self.coef_
        else:
            return pd.DataFrame({"psupertime_weight": self.coef_},
                                index=anndata.var.index.copy())


class BaselineSGDModel(PsupertimeBaseModel):
    """
    Vanilla SGDClassifier wrapper derived from `PsupertimeBaseModel`

    """
    def __init__(self,
                 method="proportional",
                 max_iter=100, 
                 random_state=1234, 
                 regularization=0.01, 
                 n_iter_no_change=5, 
                 early_stopping=True,
                 tol=1e-3,
                 learning_rate="optimal",
                 eta0=0,
                 loss='log_loss', 
                 penalty='elasticnet', 
                 l1_ratio=1, 
                 fit_intercept=True, 
                 shuffle=True, 
                 verbose=0, 
                 epsilon=0.1, 
                 n_jobs=1, 
                 power_t=0.5, 
                 validation_fraction=0.1,
                 class_weight=None,
                 warm_start=False,
                 average=False):
        
        self.method = method

        # SGD parameters
        self.eta0 = eta0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.regularization = regularization
        self.loss = loss
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.power_t = power_t
        self.validation_fraction = validation_fraction
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = average
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        
    def _binary_estimator_factory(self):
        return SGDClassifier(eta0 = self.eta0,
                            learning_rate = self.learning_rate,
                            max_iter = self.max_iter,
                            random_state = self.random_state,
                            alpha = self.regularization,
                            loss = self.loss,
                            penalty = self.penalty,
                            l1_ratio = self.l1_ratio,
                            fit_intercept = self.fit_intercept,
                            shuffle = self.shuffle,
                            verbose = self.verbose,
                            epsilon = self.epsilon,
                            n_jobs = self.n_jobs,
                            power_t = self.power_t,
                            validation_fraction = self.validation_fraction,
                            class_weight = self.class_weight,
                            warm_start = self.warm_start,
                            average = self.average,
                            early_stopping = self.early_stopping,
                            n_iter_no_change = self.n_iter_no_change,
                            tol = self.tol)


class BatchSGDModel(PsupertimeBaseModel):
    """
    BatchSGDModel is a classifier derived from `PsupertimBaseModel` that wraps an `SGDClassifier`
    as logistic binary estimator.
    
    It overwrites the superclass `_binary_estimator_factory() and `fit()` methods. The latter is wrapping
    the `SGDClassifier.partial_fit()` function to fit the model in batches for a reduced memory footprint.
    
    """
    def __init__(self,
                 method="proportional",
                 early_stopping_batches=False,
                 n_batches=1,
                 max_iter=1000, 
                 random_state=1234, 
                 regularization=0.01, 
                 n_iter_no_change=5, 
                 early_stopping=True,
                 tol=1e-3,
                 learning_rate="optimal",
                 eta0=0,
                 loss='log_loss', 
                 penalty='elasticnet', 
                 l1_ratio=0.75, 
                 fit_intercept=True, 
                 shuffle=True, 
                 verbosity=0, 
                 epsilon=0.1, 
                 n_jobs=1, 
                 power_t=0.5, 
                 validation_fraction=0.1,
                 class_weight=None,
                 warm_start=False,
                 average=False):

        self.method = method

        # model hyperparameters
        self.eta0 = eta0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.regularization = regularization
        self.loss = loss
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.verbosity = verbosity
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.power_t = power_t
        self.validation_fraction = validation_fraction
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = average
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.n_batches = n_batches
        self.early_stopping_batches = early_stopping_batches

        # data attributes
        self.k_ = None
        self.intercept_ = []
        self.coef_ = []

    def _binary_estimator_factory(self):
        return SGDClassifier(eta0 = self.eta0,
                            learning_rate = self.learning_rate,
                            max_iter = self.max_iter,
                            random_state = self.random_state,
                            alpha = self.regularization,
                            loss = self.loss,
                            penalty = self.penalty,
                            l1_ratio = self.l1_ratio,
                            fit_intercept = self.fit_intercept,
                            shuffle = self.shuffle,
                            verbose = self.verbosity >= 3,
                            epsilon = self.epsilon,
                            n_jobs = self.n_jobs,
                            power_t = self.power_t,
                            validation_fraction = self.validation_fraction,
                            class_weight = self.class_weight,
                            warm_start = self.warm_start,
                            average = self.average,
                            early_stopping = False,  # has to be false to use partial_fit
                            n_iter_no_change = self.n_iter_no_change,
                            tol = self.tol)

    def fit(self, X_org, y_org, sample_weight=None):
        """Fit ordinal logistic model. 
        Multiclass data is converted to binarized representation and one weight per feature, 
        as well as a threshold for each class is fitted with a binary logistic classifier.

        Derived from a `sklearn.linear.SGDClassifier`, fitted in batches according to `self.n_batches` 
        for reduced memory usage.
        

        :param X: Data as 2d-matrix
        :type X: numpy.array or scipy.sparse
        :param y: ordinal labels
        :type y: Iterable
        :param sample_weight: Label weights for fitting and scoring, defaults to None. Can be used for example for class balancing.
        :type sample_weight: Iterable, optional
        :return: fitted classifier
        :rtype: BatchSGDModel
        """
        rng = np.random.default_rng(self.random_state)
        X_org, y_org = self._before_fit(X_org, y_org)
        from collections import Counter
        Counter(y_org)
        if self.early_stopping:
            # TODO: This is a full copy of the input data -> split an index array instead and work with slices?
            X, X_test, y, y_test = train_test_split(X_org, y_org, test_size=self.validation_fraction,
                                                    stratify=y_org, random_state=rng.integers(9999))
            # 2023-09-18 11:32:47 awa: make the same of the number of class in train and test set
            if len(np.unique(y))!=len(np.unique(y_test)):
                # check classes
                unique_classes = np.unique(y_org)
                # iter all classes
                for class_label in unique_classes:
                    # if one classes exist in y, but not in y_test
                    if class_label not in y_test:
                        # find one sample from y and add to y_test and X_test
                        # print(class_label)
                        sample_index = np.where(y_org == class_label)[0][0]
                        y_test = np.append(y_test, y_org[sample_index])
                        X_test = np.vstack((X_test, X_org[sample_index]))

            # TODO: initializing binarized matrices for testing can be significant memory sink!
            y_test_bin = restructure_y_to_bin(y_test)
            # del(y_test)

            if self.early_stopping_batches:
                n_test = X_test.shape[0]
                test_indices = np.arange(len(y_test_bin))
            else:

                # X_test_bin = restructure_X_to_bin(X_test, len(np.unique(y_test))-1)
                X_test_bin = restructure_X_to_bin(X_test, self.k_)
                # del(X_test)
        
        # diagonal matrix, to construct the binarized X per batch
        thresholds = np.identity(self.k_)
        if sparse.issparse(X):
            thresholds = sparse.csr_matrix(thresholds)

        model = self.get_binary_estimator()
        n = X.shape[0]

        # binarize only the labels already
        y_bin = restructure_y_to_bin(y)

        # create an inex array and shuffle
        sampled_indices = rng.integers(len(y_bin), size=len(y_bin))

        # iterations over all data
        epoch = 0

        # tracking previous scores for early stopping
        best_score = - np.inf
        n_no_improvement = 0

        while epoch < self.max_iter:

            epoch += 1

            start = 0
            for i in range(1, self.n_batches+1):
                end = (i * len(y_bin) // self.n_batches)
                batch_idx = sampled_indices[start:end]
                batch_idx_mod_n = batch_idx % n
                
                if sparse.issparse(X):
                    X_batch = sparse.hstack((X[batch_idx_mod_n], thresholds[batch_idx // n]))
                else:
                    X_batch = np.hstack((X[batch_idx_mod_n,:], thresholds[batch_idx // n]))
                
                y_batch = y_bin[batch_idx]
                start = end
                weights = np.array(sample_weight)[batch_idx_mod_n] if sample_weight is not None else None
                model.partial_fit(X_batch, y_batch, classes=np.unique(y_batch), sample_weight=weights)

            # Early stopping using the test data 
            if self.early_stopping:

                # build test data in batches as needed to avoid keeping in memory
                if self.early_stopping_batches:
                    scores = []
                    start = 0
                    for i in range(1, self.n_batches+1):
                        end = (i * len(y_test_bin) // self.n_batches)
                        batch_idx = test_indices[start:end]
                        batch_idx_mod_n = batch_idx % n_test
                        if sparse.issparse(X_test):
                            X_test_batch = sparse.hstack((X_test[batch_idx_mod_n], thresholds[batch_idx // n_test]))
                        else:
                            X_test_batch = np.hstack((X_test[batch_idx_mod_n], thresholds[batch_idx // n_test]))
                        
                        scores.append(model.score(X_test_batch, y_test_bin[batch_idx]))
                        start = end          
                        
                    cur_score = np.mean(scores)
                
                else:
                    cur_score = model.score(X_test_bin, y_test_bin)

                if cur_score - self.tol > best_score:
                    best_score = cur_score
                    n_no_improvement = 0
                else:
                    n_no_improvement += 1
                    if n_no_improvement >= self.n_iter_no_change:
                        if self.verbosity >= 2:
                            print("Stopped early at epoch ", epoch, " Current score:", cur_score)
                        break

            if self.shuffle:
                sampled_indices = rng.integers(len(y_bin), size=len(y_bin))

            # TODO: Learning Rate adjustments?

        self._after_fit(model)
        return self
