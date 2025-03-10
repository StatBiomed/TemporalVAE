import numpy as np
from typing import Iterable
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator, clone
from copy import copy
import warnings


class RegularizationSearchCV:

    def __init__(self,
                 estimator: BaseEstimator,
                 scoring="accuracy",
                 reg_param_name="regularization",
                 reg_path=None,
                 n_params=40,
                 reg_high=1,
                 reg_low=0.001,
                 n_jobs=-1,
                 n_folds=5,
                 verbosity=1):

        self.verbosity = verbosity
        self.is_fitted = False
        self.n_jobs = n_jobs
        self.n_folds = n_folds

        self.scoring = scoring
        if not isinstance(scoring, str) and (isinstance(scoring, dict) or isinstance(scoring, Iterable)):
            warnings.warn(
                "Parameter 'scoring' is a list or dict: Multiple scorers are currently not supported. Using accuracy for now.")
            self.scoring = "accuracy"

        if reg_path is None:
            self.reg_path = np.geomspace(reg_high, reg_low, n_params)
        else:
            try:
                self.reg_path = np.array(reg_path).astype("float")
            except ValueError as e:
                print(e)
                raise ValueError("Parameter 'reg_path' is not Iterable or cannot be converted to float")

        self.estimator = estimator
        try:
            est_instance = self.estimator()
        except TypeError as e:
            print(e)
            raise ValueError("Parameter 'estimator' could not be initiated. Did you pass an instance?")

        if not isinstance(est_instance, BaseEstimator):
            raise ValueError("Parameter 'estimator' is not a sklearn.base.BaseEstimator")

        self.reg_param_name = reg_param_name
        if not hasattr(est_instance, self.reg_param_name):
            raise ValueError("Parameter 'reg_param_name' is not a valid parameter for %s" % self.estimator.__name__)

            # average cross validation scores for each lambda
        self.scores = []

        # std of cross validation scores for each lambda
        self.scores_std = []

        # average training scores fore each lambda
        self.train_scores = []

        # std of training scores fore each lambda
        self.train_scores_std = []

        # best degrees of freedom
        self.dof = []

        # best estimators
        self.fitted_estimators = []

    def fit(self, X, y, fit_params=dict(), estimator_params=dict()):

        # copy the params, in order to not mutate the original object
        estimator_params = copy(estimator_params)

        if not isinstance(estimator_params, dict):
            raise ValueError("estimator_params must be of type dict")

        if not isinstance(fit_params, dict):
            raise ValueError("fit_params must be of type dict")

        for i, reg in enumerate(self.reg_path):

            if self.verbosity >= 1:
                print("Regularization: %s/%s" % (i + 1, len(self.reg_path)), sep="", end="\r")

            estimator_params[self.reg_param_name] = reg
            cv = cross_validate(estimator=self.estimator(**estimator_params),
                                scoring=self.scoring,
                                n_jobs=self.n_jobs,
                                cv=self.n_folds,
                                X=X,
                                y=y,
                                error_score="raise",
                                return_train_score=True,
                                return_estimator=True,
                                fit_params=fit_params
                                )

            best_idx = np.argmax(cv["test_score"])
            self.train_scores.append(np.mean(cv["train_score"]))
            self.train_scores_std.append(np.std(cv["train_score"]))
            self.scores.append(np.mean(cv["test_score"]))
            self.scores_std.append(np.std(cv["test_score"]))
            self.fitted_estimators.append(cv["estimator"][best_idx])

            # TODO: Disregard the threshold weights when running grid search on a binary model
            weights = np.array(cv["estimator"][best_idx].coef_).flatten()
            self.dof.append(np.count_nonzero(weights))

        if self.verbosity >= 1:
            print("Regularization: done   ")

        self.is_fitted_ = True
        return self

    def get_optimal_regularization(self, method="1se", index=None):
        """Returns a tuple of (idx, reg_param) where reg_param is the optimal regularization value
        according to the chosen method and idx its iteration step in the search.

        Note: if method is "1se", but only a zero degree-of-freedom model can be found,
        it automatically retries with method "best".


        :param method: Specify the method by which optimality is determined. Must be one of {"1se", "half_se", "best", "index"} defaults to "1se"
        :type method: str, optional
        :param index: Model at specific index that should be returned.
         Only if method="index" is selected, defaults to None
        :type index: integer, optional
        :raises ValueError: When an invalid method parameter is given
        :raises ValueError: When method "index" is chosen and the given index is out of bounds
        :return: tuple of (idx, optimal_param)
        :rtype: tuple
        """

        if not method in ["1se", "half_se", "best", "index"]:
            raise ValueError("The method parameter should be one of '1se' or 'best'")

        if method == "index":
            if index is None or (index >= len(self.scores) or index < 0):
                raise ValueError("Parmeter `index` must be set to a valid cv index, if method='index' is selected.")
            return (self.reg_path[index], index)

        if method == "best":
            idx = np.argmax(self.scores)
            return (self.reg_path[idx], idx)

        if method in ["1se", "half_se"]:
            n = len(self.dof)

            # check the effect direction of the regularization parameter
            sparsity_increases_w_idx = np.mean(self.dof[:n // 4]) < np.mean(self.dof[-n // 4:])

            # compute the threshold as the maximum score minus the standard error
            nonzero_idx = np.nonzero(self.dof)
            max_idx = np.argmax(self.scores)

            if method == "1se":
                tol = np.std(np.array(self.scores)[nonzero_idx])

            else:  # method == "half_se"
                tol = 0.5 * np.std(np.array(self.scores)[nonzero_idx])

            thresh = self.scores[max_idx] - tol

            if sparsity_increases_w_idx:
                items = zip(self.scores, self.dof)
            else:
                items = reversed(list(zip(self.scores, self.dof)))

            for i, (s, d) in enumerate(items):
                # exclude models with 0 degrees of freedom
                # and stop if there is no sufficiently good sparser model
                if (s > thresh and d != 0) or \
                        (i == max_idx):
                    return (self.reg_path[i], i)

            warnings.warn(
                "No model for method '%s' with non-zero degrees of freedom could be found. Returning the best scoring model" % method)
            return self.get_optimal_regularization(method="best")

    def get_optimal_parameters(self, *args, **kwargs):
        """ Returns the parameters of model with optimal reg_param. See `get_optimal_regularization` for more details

        :return: Model parameters
        :rtype: dict
        """
        reg, idx = self.get_optimal_regularization(*args, **kwargs)
        return self.fitted_estimators[idx].get_params()

    def get_optimal_model(self, *args, **kwargs):
        """ Returns the model with optimal reg_param. See `get_optimal_regularization` for more details

        :return: Optimal model
        :rtype: sklearn.base.BaseModel
        """
        return self.estimator(**self.get_optimal_parameters(*args, **kwargs))
