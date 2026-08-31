"""Classical baselines on temporally pooled landmark features.

These exist because they are the control the sign-recognition literature
usually omits. A recurrent network on 100 training sequences has far more
capacity than the data can constrain; if a random forest on pooled
statistics matches it, the recurrent architecture is not the thing producing
the accuracy, and a paper that reports only the LSTM has mis-attributed its
own result.
"""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..features import pool_sequence, pyramid_pool
from .base import SequenceClassifier


class PooledClassical(SequenceClassifier):
    """Wraps any scikit-learn estimator behind the sequence interface.

    Sequences are reduced with temporal pyramid pooling by default, which
    preserves coarse ordering. Without it these models cannot distinguish a
    sign from its time-reverse, and several sign pairs differ in exactly that
    way.
    """

    kind = "classical"

    def __init__(self, n_classes: int, seed: int = 0,
                 pooling: tuple[str, ...] = ("mean", "std", "min", "max"),
                 pyramid_levels: tuple[int, ...] = (1, 2, 4),
                 **kwargs) -> None:
        super().__init__(n_classes, seed, **kwargs)
        self.pooling = pooling
        self.pyramid_levels = pyramid_levels
        self.estimator = self._build()

    def _build(self):
        raise NotImplementedError

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            if self.pyramid_levels:
                return pyramid_pool(X, self.pyramid_levels, self.pooling)
            return pool_sequence(X, self.pooling)
        return X

    def _fit(self, X: np.ndarray, y: np.ndarray,
             X_val: np.ndarray | None = None,
             y_val: np.ndarray | None = None) -> None:
        # Classical estimators here do not early-stop, so the validation fold
        # is simply unused rather than silently folded into training.
        self.estimator.fit(self._prepare(X), y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.estimator.predict_proba(self._prepare(X))
        # Guard against folds where a class is absent from training data.
        if p.shape[1] != self.n_classes:
            full = np.zeros((p.shape[0], self.n_classes), dtype=np.float64)
            classes = getattr(self.estimator, "classes_", None)
            if classes is None and hasattr(self.estimator, "steps"):
                classes = self.estimator.steps[-1][1].classes_
            full[:, np.asarray(classes, dtype=int)] = p
            return full
        return p

    def n_parameters(self) -> int:
        est = self.estimator
        if hasattr(est, "steps"):
            est = est.steps[-1][1]
        if hasattr(est, "coefs_"):
            return int(sum(c.size for c in est.coefs_)
                       + sum(b.size for b in est.intercepts_))
        if hasattr(est, "estimators_"):
            return int(sum(t.tree_.node_count for t in est.estimators_))
        if hasattr(est, "coef_"):
            return int(np.asarray(est.coef_).size + np.asarray(est.intercept_).size)
        return -1

    def model_size_bytes(self) -> int:
        return len(pickle.dumps(self.estimator))


class RandomForestModel(PooledClassical):
    name = "random_forest"

    def _build(self):
        return RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 300),
            max_depth=self.params.get("max_depth", None),
            min_samples_leaf=self.params.get("min_samples_leaf", 1),
            random_state=self.seed,
            n_jobs=-1,
        )


class ExtraTreesModel(PooledClassical):
    name = "extra_trees"

    def _build(self):
        return ExtraTreesClassifier(
            n_estimators=self.params.get("n_estimators", 300),
            random_state=self.seed,
            n_jobs=-1,
        )


class SVMModel(PooledClassical):
    name = "svm_rbf"

    def _build(self):
        return Pipeline([
            ("scale", StandardScaler()),
            ("svc", SVC(
                C=self.params.get("C", 10.0),
                gamma=self.params.get("gamma", "scale"),
                kernel="rbf",
                probability=True,
                random_state=self.seed,
            )),
        ])


class LogRegModel(PooledClassical):
    name = "logreg"

    def _build(self):
        return Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(
                C=self.params.get("C", 1.0),
                max_iter=self.params.get("max_iter", 2000),
                random_state=self.seed,
            )),
        ])


class KNNModel(PooledClassical):
    name = "knn"

    def _build(self):
        return Pipeline([
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=self.params.get("n_neighbors", 5),
            )),
        ])


class MLPModel(PooledClassical):
    """The static pathway's classifier, applied to pooled sequences.

    Included so the fingerspelling model and the word-sign models appear in
    the same table under the same protocol.
    """

    name = "mlp"

    def _build(self):
        return Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=self.params.get("hidden_layer_sizes", (256, 128)),
                max_iter=self.params.get("max_iter", 800),
                early_stopping=self.params.get("early_stopping", False),
                random_state=self.seed,
            )),
        ])


CLASSICAL = {
    m.name: m
    for m in (RandomForestModel, ExtraTreesModel, SVMModel,
              LogRegModel, KNNModel, MLPModel)
}
