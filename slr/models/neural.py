"""Neural sequence models.

Includes ``lstm_original`` - a faithful reproduction of the architecture in
the original ``train_model.py`` - so the paper can report the published-style
baseline and the corrected models in the same table.

Two defects in that original are fixed in every other model here:

* ``LSTM(..., activation='relu')``. ReLU as an LSTM cell activation is
  unbounded and the recurrent state diverges; the original training logs show
  loss spiking to 23.6, 8.2 and 7.6 across runs. The default ``tanh`` is
  bounded and stable.
* No seed control. The same script produced final training accuracies from
  0.41 to 0.87 across six logged runs. Every model here seeds Python, NumPy
  and TensorFlow, and the driver repeats each configuration so results are
  reported as mean +/- std rather than as whichever run looked best.
"""

from __future__ import annotations

import os
import random

import numpy as np

from .base import SequenceClassifier

# Silence TF's C++ logging before it is imported anywhere.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _tf():
    """Import TensorFlow lazily so the package is usable without it."""
    import tensorflow as tf
    return tf


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        _tf().random.set_seed(seed)
    except Exception:
        pass


class KerasSequenceModel(SequenceClassifier):
    """Shared training loop for the Keras architectures."""

    kind = "neural"

    def __init__(self, n_classes: int, seed: int = 0, epochs: int = 200,
                 batch_size: int = 16, patience: int = 25,
                 learning_rate: float = 1e-3, verbose: int = 0, **kwargs) -> None:
        super().__init__(n_classes, seed, epochs=epochs, batch_size=batch_size,
                         patience=patience, learning_rate=learning_rate, **kwargs)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.model = None
        self.history = None

    # ------------------------------------------------------------ subclass

    def _build(self, input_shape: tuple[int, int]):
        raise NotImplementedError

    # ---------------------------------------------------------------- fit

    def _fit(self, X, y, X_val=None, y_val=None) -> None:
        tf = _tf()
        set_global_seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y_cat = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

        self.model = self._build(X.shape[1:])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = []
        validation_data = None
        if X_val is not None and len(X_val) > 0:
            validation_data = (
                np.asarray(X_val, dtype=np.float32),
                tf.keras.utils.to_categorical(y_val, num_classes=self.n_classes),
            )
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience,
                restore_best_weights=True,
            ))
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=max(5, self.patience // 3), min_lr=1e-5, verbose=0,
            ))

        self.history = self.model.fit(
            X, y_cat,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=self.verbose,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X, dtype=np.float32), verbose=0)

    def n_parameters(self) -> int:
        if self.model is None:
            return -1
        return int(sum(np.prod(w.shape) for w in self.model.trainable_weights))

    def model_size_bytes(self) -> int:
        # float32 weights; the deployable footprint before any quantisation.
        return self.n_parameters() * 4


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class LSTMOriginal(KerasSequenceModel):
    """Faithful reproduction of the original ``train_model.py`` network.

    LSTM(64, activation='relu') -> Dropout(0.2) -> Dense(32) -> Dense(16)
    -> softmax. Reported as the paper's "prior configuration" row, defects
    included, so the effect of correcting them is measurable rather than
    asserted.
    """

    name = "lstm_original"

    def _build(self, input_shape):
        tf = _tf()
        L = tf.keras.layers
        return tf.keras.Sequential([
            L.Input(shape=input_shape),
            L.LSTM(64, return_sequences=False, activation="relu"),
            L.Dropout(0.2),
            L.Dense(32, activation="relu"),
            L.Dense(16, activation="relu"),
            L.Dense(self.n_classes, activation="softmax"),
        ])


class LSTMModel(KerasSequenceModel):
    name = "lstm"

    def _build(self, input_shape):
        tf = _tf()
        L = tf.keras.layers
        units = self.params.get("units", 64)
        return tf.keras.Sequential([
            L.Input(shape=input_shape),
            L.LSTM(units, return_sequences=False),       # tanh: bounded, stable
            L.Dropout(self.params.get("dropout", 0.3)),
            L.Dense(32, activation="relu"),
            L.Dense(self.n_classes, activation="softmax"),
        ])


class BiLSTMModel(KerasSequenceModel):
    name = "bilstm"

    def _build(self, input_shape):
        tf = _tf()
        L = tf.keras.layers
        units = self.params.get("units", 48)
        return tf.keras.Sequential([
            L.Input(shape=input_shape),
            L.Bidirectional(L.LSTM(units, return_sequences=False)),
            L.Dropout(self.params.get("dropout", 0.3)),
            L.Dense(32, activation="relu"),
            L.Dense(self.n_classes, activation="softmax"),
        ])


class GRUModel(KerasSequenceModel):
    name = "gru"

    def _build(self, input_shape):
        tf = _tf()
        L = tf.keras.layers
        units = self.params.get("units", 64)
        return tf.keras.Sequential([
            L.Input(shape=input_shape),
            L.GRU(units, return_sequences=False),
            L.Dropout(self.params.get("dropout", 0.3)),
            L.Dense(32, activation="relu"),
            L.Dense(self.n_classes, activation="softmax"),
        ])


class CNN1DModel(KerasSequenceModel):
    """Temporal convolution. Far cheaper than a recurrent pass and a strong
    baseline on short fixed-length sequences."""

    name = "cnn1d"

    def _build(self, input_shape):
        tf = _tf()
        L = tf.keras.layers
        f = self.params.get("filters", 64)
        return tf.keras.Sequential([
            L.Input(shape=input_shape),
            L.Conv1D(f, 5, padding="same", activation="relu"),
            L.BatchNormalization(),
            L.MaxPooling1D(2),
            L.Conv1D(f * 2, 3, padding="same", activation="relu"),
            L.BatchNormalization(),
            L.GlobalAveragePooling1D(),
            L.Dropout(self.params.get("dropout", 0.3)),
            L.Dense(64, activation="relu"),
            L.Dense(self.n_classes, activation="softmax"),
        ])


class TransformerModel(KerasSequenceModel):
    """Single-block transformer encoder with learned positional embeddings."""

    name = "transformer"

    def _build(self, input_shape):
        tf = _tf()
        L = tf.keras.layers
        T, F = input_shape
        d_model = self.params.get("d_model", 64)
        heads = self.params.get("num_heads", 4)
        ff = self.params.get("ff_dim", 128)
        drop = self.params.get("dropout", 0.2)

        inp = L.Input(shape=input_shape)
        x = L.Dense(d_model)(inp)
        pos = L.Embedding(input_dim=T, output_dim=d_model)(tf.range(T))
        x = x + pos

        attn = L.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads,
                                    dropout=drop)(x, x)
        x = L.LayerNormalization(epsilon=1e-6)(x + attn)
        ffn = L.Dense(ff, activation="relu")(x)
        ffn = L.Dense(d_model)(ffn)
        x = L.LayerNormalization(epsilon=1e-6)(x + ffn)

        x = L.GlobalAveragePooling1D()(x)
        x = L.Dropout(drop)(x)
        out = L.Dense(self.n_classes, activation="softmax")(x)
        return tf.keras.Model(inp, out)


NEURAL = {
    m.name: m
    for m in (LSTMOriginal, LSTMModel, BiLSTMModel, GRUModel,
              CNN1DModel, TransformerModel)
}
