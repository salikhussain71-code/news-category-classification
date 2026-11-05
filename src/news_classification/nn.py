"""Bonus: simple feed-forward neural network (Keras) on TF-IDF features.

TF-IDF matrices are huge when dense (documents x 50,000 columns), so mini-batches are densified
on the fly. A plain training loop is used (instead of ``fit``) for compatibility across Keras 2/3.
"""

from __future__ import annotations

import os
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from . import config  # noqa: E402


def _import_keras():
    try:
        import keras
    except ImportError as err:
        raise ImportError(
            "The neural-network bonus needs TensorFlow/Keras: pip install -r requirements-extra.txt"
        ) from err
    return keras


def build_network(input_dim: int, n_classes: int, hidden: tuple[int, int] = (256, 128)):
    keras = _import_keras()
    layers = keras.layers
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(hidden[0], activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(hidden[1], activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _batches(X, indices, batch_size):
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        yield idx, X[idx].toarray().astype("float32")


def predict_proba(model, X, batch_size: int = 512) -> np.ndarray:
    out = [model.predict_on_batch(xb) for _, xb in _batches(X, np.arange(X.shape[0]), batch_size)]
    return np.vstack([np.asarray(o) for o in out])


def train_network(X_train, y_train, X_val, y_val, epochs: int = 8, batch_size: int = 256,
                  patience: int = 2, seed: int = config.RANDOM_STATE, verbose: bool = True):
    """Train with early stopping on validation loss; returns ``(model, history, fit_seconds)``."""
    keras = _import_keras()
    keras.utils.set_random_seed(seed)
    y_train, y_val = np.asarray(y_train), np.asarray(y_val)
    model = build_network(X_train.shape[1], int(max(y_train.max(), y_val.max())) + 1)

    rng = np.random.default_rng(seed)
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_loss, best_weights, wait = np.inf, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(X_train.shape[0])
        losses, accs, sizes = [], [], []
        for idx, xb in _batches(X_train, order, batch_size):
            loss, acc = model.train_on_batch(xb, y_train[idx])
            losses.append(float(loss)); accs.append(float(acc)); sizes.append(len(idx))  # noqa: E702
        val_prob = predict_proba(model, X_val)
        val_loss = float(-np.log(np.clip(val_prob[np.arange(len(y_val)), y_val], 1e-12, None)).mean())
        val_acc = float((val_prob.argmax(axis=1) == y_val).mean())
        history["loss"].append(float(np.average(losses, weights=sizes)))
        history["accuracy"].append(float(np.average(accs, weights=sizes)))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        if verbose:
            print(f"  epoch {epoch:2d}/{epochs} loss={history['loss'][-1]:.4f} "
                  f"acc={history['accuracy'][-1]:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_loss - 1e-4:
            best_loss, best_weights, wait = val_loss, model.get_weights(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_weights is not None:
        model.set_weights(best_weights)
    return model, history, time.perf_counter() - t0
