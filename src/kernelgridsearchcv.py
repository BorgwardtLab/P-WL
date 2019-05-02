from sklearn.model_selection import ParameterGrid, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import numpy as np

class KernelGridSearchCV:
    """
    A simple class for performing a grid search for kernel matrices with
    a cross-validation strategy. At present, the class interface follows
    the default interface of `scikit-learn`. However, the class is *not*
    yet inheriting from any base class.
    """

    def __init__(self, clf, param_grid, cv=None, random_state=None, refit=True):
        self._clf = clf
        self._grid = param_grid
        self._cv = cv
        self._random_state = random_state
        self._refit = refit
        self._best_estimator = None
        self._best_score = None

    def fit(self, X, y):
        if self._cv is None:
            cv = KFold(n_splits=3, shuffle=True, random_state=self._random_state)
        elif isinstance(self._cv, int):
            cv = StratifiedKFold(n_splits=self._cv, shuffle=True, random_state=self.random_state)
        else:
            cv = self._cv

        grid = ParameterGrid(self._grid)

        for parameters in grid:
            clf = self._clf
            clf.set_params(**parameters)

            scores = []
            for train, test in cv.split(np.zeros(len(y)), y):
                X_train = X[train][:, train]
                y_train = y[train]
                X_test = X[test][:, train]
                y_test = y[test]

                clf.fit(X_train, y_train)

                ap = accuracy_score(y_test, clf.predict(X_test))
                scores.append(ap)

            score = np.mean(scores)
            if self._best_score is None or score > self._best_score:
                self._best_estimator = clone(clf)
                self._best_score = score
                self._best_params = parameters
