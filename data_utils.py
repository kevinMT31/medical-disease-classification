from collections import Counter

import numpy as np
from scipy.io import arff


def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    if isinstance(value, str):
        return value.strip()
    return value


def clean_label(value):
    value = decode_value(value)

    if isinstance(value, (int, float, np.number)):
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))

    return str(value).strip().lower()


def to_float(value):
    value = decode_value(value)

    if value is None or value in {"", "?"}:
        return np.nan

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Feature value {value!r} is not numeric.") from error

    return number if np.isfinite(number) else np.nan


def to_category(value):
    value = decode_value(value)

    if value is None or value in {"", "?"}:
        return None

    return str(value)


def encode_diabetes_labels(raw_labels):
    negative = {"0", "negative", "tested_negative", "no", "false"}
    positive = {"1", "positive", "tested_positive", "yes", "true"}
    labels = []

    for raw_label in raw_labels:
        label = clean_label(raw_label)

        if label in negative:
            labels.append(0)
        elif label in positive:
            labels.append(1)
        else:
            raise ValueError(f"Unknown diabetes label: {raw_label!r}")

    return np.asarray(labels, dtype=np.int64)


def encode_breast_cancer_labels(raw_labels):
    labels = [clean_label(value) for value in raw_labels]
    unique_labels = set(labels)

    if unique_labels <= {"benign", "malignant"}:
        mapping = {"benign": 0, "malignant": 1}
    elif unique_labels <= {"0", "1"}:
        mapping = {"0": 0, "1": 1}
    elif unique_labels <= {"1", "2"}:
        mapping = {"1": 0, "2": 1}
    elif unique_labels <= {"2", "4"}:
        mapping = {"2": 0, "4": 1}
    else:
        raise ValueError(f"Unknown breast cancer labels: {sorted(unique_labels)}")

    return np.asarray([mapping[label] for label in labels], dtype=np.int64)


def encode_hypothyroid_labels(raw_labels):
    labels = [clean_label(value) for value in raw_labels]
    positive_labels = {
        "compensated_hypothyroid",
        "primary_hypothyroid",
        "secondary_hypothyroid",
    }
    unknown_labels = set(labels) - (positive_labels | {"negative"})

    if unknown_labels:
        raise ValueError(f"Unknown hypothyroid labels: {sorted(unknown_labels)}")

    # Hypothyroid has three positive classes, grouped here into one positive class.
    return np.asarray(
        [int(label in positive_labels) for label in labels],
        dtype=np.int64,
    )


def load_dataset(path, dataset_key):
    data, metadata = arff.loadarff(str(path))
    columns = list(metadata.names())
    column_types = [
        "numeric"
        if value_type in {"numeric", "real", "integer"}
        else "categorical"
        for value_type in metadata.types()
    ]

    if len(columns) < 2:
        raise ValueError("The dataset needs feature and class columns.")

    feature_columns = columns[:-1]
    feature_types = column_types[:-1]

    # Some Breast Cancer Wisconsin files include an ID before the nine features.
    if dataset_key == "breast_cancer" and len(feature_columns) == 10:
        feature_columns = feature_columns[1:]
        feature_types = feature_types[1:]

    if dataset_key == "breast_cancer" and len(feature_columns) != 9:
        raise ValueError(
            "Expected 9 breast cancer features and an optional ID column."
        )

    X = np.empty((len(data), len(feature_columns)), dtype=object)

    for index, (name, feature_type) in enumerate(
        zip(feature_columns, feature_types)
    ):
        if feature_type == "numeric":
            X[:, index] = [to_float(value) for value in data[name]]
        else:
            X[:, index] = [to_category(value) for value in data[name]]

    raw_labels = data[columns[-1]]

    if dataset_key == "diabetes":
        y = encode_diabetes_labels(raw_labels)
    elif dataset_key == "breast_cancer":
        y = encode_breast_cancer_labels(raw_labels)
    elif dataset_key == "hypothyroid":
        y = encode_hypothyroid_labels(raw_labels)
    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    return X, y, feature_types


def encode_features(X, feature_steps):
    encoded_columns = []

    for step in feature_steps:
        feature_type, column_index = step[0], step[1]

        if feature_type == "numeric":
            median = step[2]
            column = np.asarray(X[:, column_index], dtype=np.float64)
            column = np.where(np.isnan(column), median, column)
            encoded_columns.append(column.reshape(-1, 1))
        else:
            mode, categories = step[2], step[3]
            values = [
                value if value is not None else mode
                for value in X[:, column_index]
            ]
            category_indices = {
                category: index for index, category in enumerate(categories)
            }
            one_hot = np.zeros(
                (len(values), len(categories)),
                dtype=np.float64,
            )

            for row_index, value in enumerate(values):
                if value in category_indices:
                    one_hot[row_index, category_indices[value]] = 1.0

            encoded_columns.append(one_hot)

    return np.hstack(encoded_columns)


def fit_preprocessor(X_train, feature_types):
    feature_steps = []

    for column_index, feature_type in enumerate(feature_types):
        if feature_type == "numeric":
            column = np.asarray(X_train[:, column_index], dtype=np.float64)
            available = column[~np.isnan(column)]
            median = float(np.median(available)) if available.size else 0.0
            feature_steps.append(("numeric", column_index, median))
        else:
            available = [
                value
                for value in X_train[:, column_index]
                if value is not None
            ]
            mode = (
                Counter(available).most_common(1)[0][0]
                if available
                else "missing"
            )
            filled = [
                value if value is not None else mode
                for value in X_train[:, column_index]
            ]
            categories = sorted(set(filled))
            feature_steps.append(
                ("categorical", column_index, mode, categories)
            )

    X_encoded = encode_features(X_train, feature_steps)
    means = np.mean(X_encoded, axis=0)
    stds = np.std(X_encoded, axis=0)
    stds = np.where(stds == 0, 1.0, stds)

    return feature_steps, means, stds


def preprocess(X, preprocessor):
    feature_steps, means, stds = preprocessor
    X_encoded = encode_features(X, feature_steps)
    return ((X_encoded - means) / stds).astype(np.float32)
