from pathlib import Path

from data_utils import load_dataset
from evaluation import evaluate_dataset, print_final_summary, set_seed


SEED = 2025
FOLDS = 5
EPOCHS = 300
LEARNING_RATE = 0.01

BASE_DIR = Path(__file__).resolve().parent

DATASETS = [
    ("Diabetes", BASE_DIR / "data" / "diabetes.arff", "diabetes"),
    (
        "Breast Cancer Wisconsin",
        BASE_DIR / "data" / "breast-cancer-wisconsin.arff",
        "breast_cancer",
    ),
    ("Hypothyroid", BASE_DIR / "data" / "hypothyroid.arff", "hypothyroid"),
]


def main():
    set_seed(SEED)
    print("Educational classification project, not medical advice.")

    all_results = {}

    for dataset_name, path, dataset_key in DATASETS:
        if not path.is_file():
            print(
                f"Skipping {dataset_name}: "
                f"file not found at {path.relative_to(BASE_DIR)}"
            )
            continue

        try:
            X, y, feature_types = load_dataset(path, dataset_key)
            all_results[dataset_name] = evaluate_dataset(
                X=X,
                y=y,
                feature_types=feature_types,
                dataset_name=dataset_name,
                seed=SEED,
                folds=FOLDS,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
            )
        except (OSError, ValueError) as error:
            print(f"Skipping {dataset_name}: {error}")

    if all_results:
        print_final_summary(all_results)
    else:
        print("\nNo datasets were available.")


if __name__ == "__main__":
    main()
