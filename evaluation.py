import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn

from data_utils import fit_preprocessor, preprocess
from models import LogisticRegression, MLP


MODELS = {
    "Logistic Regression": LogisticRegression,
    "MLP": MLP,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model, X, y, epochs, learning_rate):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()


def predict(model, X):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_tensor)).cpu().numpy().reshape(-1)

    return (probabilities >= 0.5).astype(np.int64)


def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_dataset(
    X,
    y,
    feature_types,
    dataset_name,
    seed,
    folds,
    epochs,
    learning_rate,
):
    class_counts = np.bincount(y, minlength=2)

    if np.any(class_counts < folds):
        raise ValueError(
            f"Each class needs at least {folds} rows. "
            f"Class counts: {class_counts.tolist()}."
        )

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    scores = {
        model_name: {metric: [] for metric in metric_names}
        for model_name in MODELS
    }

    print(f"\n{dataset_name}: {len(y)} rows")

    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(X, y),
        start=1,
    ):
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        preprocessor = fit_preprocessor(X_train, feature_types)
        X_train = preprocess(X_train, preprocessor)
        X_val = preprocess(X_val, preprocessor)

        print(f"  Fold {fold}")

        for model_index, (model_name, model_class) in enumerate(MODELS.items()):
            set_seed(seed + fold + model_index * 100)
            model = model_class(X_train.shape[1])
            train_model(model, X_train, y_train, epochs, learning_rate)

            fold_metrics = get_metrics(y_val, predict(model, X_val))

            for metric, value in fold_metrics.items():
                scores[model_name][metric].append(value)

            print(f"    {model_name}: F1={fold_metrics['F1-score']:.4f}")

    return {
        model_name: {
            metric: float(np.mean(values))
            for metric, values in model_scores.items()
        }
        for model_name, model_scores in scores.items()
    }


def print_final_summary(all_results):
    print("\nFinal comparison")
    print(
        f"{'Dataset':<28} | {'Model':<27} | "
        f"{'Accuracy':>8} | {'Precision':>9} | "
        f"{'Recall':>7} | {'F1-score':>8}"
    )
    print("-" * 108)

    for dataset_name, model_results in all_results.items():
        best_model = max(
            model_results,
            key=lambda name: model_results[name]["F1-score"],
        )

        for model_name, metrics in model_results.items():
            display_name = (
                f"{model_name} (best)"
                if model_name == best_model
                else model_name
            )
            print(
                f"{dataset_name:<28} | {display_name:<27} | "
                f"{metrics['Accuracy']:>8.4f} | "
                f"{metrics['Precision']:>9.4f} | "
                f"{metrics['Recall']:>7.4f} | "
                f"{metrics['F1-score']:>8.4f}"
            )
