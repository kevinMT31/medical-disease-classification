# Medical Disease Classification

A PyTorch-based machine learning project for comparing binary classification models across three structured medical datasets: Diabetes, Breast Cancer Wisconsin, and Hypothyroid.

The project compares Logistic Regression and Multi-Layer Perceptron (MLP) models using 5-fold stratified cross-validation and standard classification metrics.

This project is for educational purposes only and is not medical advice.

## Features

* Loads medical datasets from ARFF files
* Supports Diabetes, Breast Cancer Wisconsin, and Hypothyroid datasets
* Compares PyTorch Logistic Regression and MLP models
* Uses 5-fold stratified cross-validation
* Applies leakage-safe preprocessing inside each training fold
* Handles numeric and categorical ARFF features
* Reports Accuracy, Precision, Recall, and F1-score
* Prints a final comparison table and marks the best model for each dataset

## Dataset Files

Place the dataset files inside the `data/` folder:

```text
data/diabetes.arff
data/breast-cancer-wisconsin.arff
data/hypothyroid.arff
```

If one dataset file is missing, the program will skip that dataset and continue running the available ones.

## Installation

Clone the repository:

```bash
git clone https://github.com/kevinmt31/medical-disease-classification.git
cd medical-disease-classification
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

Run the benchmark:

```bash
python main.py
```

The script automatically evaluates all available datasets and prints the final comparison table.

## Models

The project compares two PyTorch binary classification models:

### Logistic Regression

A simple linear model using one fully connected layer.

### Multi-Layer Perceptron

A small neural network with two hidden layers and ReLU activation.

Both models return logits and are trained using `BCEWithLogitsLoss`.

## Evaluation Method

Each dataset is evaluated using 5-fold stratified cross-validation.

For each fold:

1. The training and validation data are split.
2. Missing values and categorical encodings are fitted only on the training fold.
3. The same preprocessing is applied to the validation fold.
4. Logistic Regression and MLP are trained and evaluated.
5. Accuracy, Precision, Recall, and F1-score are recorded.

The best model for each dataset is selected based on average F1-score.

## Results

Example results from one run:

```text
Dataset                      | Model                       | Accuracy | Precision |  Recall | F1-score
------------------------------------------------------------------------------------------------------------
Diabetes                     | Logistic Regression (best)  |   0.7709 |    0.7239 |  0.5632 |   0.6276
Diabetes                     | MLP                         |   0.6810 |    0.5450 |  0.5444 |   0.5411
Breast Cancer Wisconsin      | Logistic Regression (best)  |   0.9657 |    0.9468 |  0.9545 |   0.9506
Breast Cancer Wisconsin      | MLP                         |   0.9600 |    0.9247 |  0.9628 |   0.9433
Hypothyroid                  | Logistic Regression         |   0.9510 |    0.9153 |  0.4020 |   0.5585
Hypothyroid                  | MLP (best)                  |   0.9788 |    0.9171 |  0.7970 |   0.8524
```

Results may vary slightly depending on package versions and hardware.

## Dataset Notice

The datasets used in this project are public educational benchmark datasets. They are included for reproducibility and learning purposes.

Please refer to and cite the original dataset sources when reusing them:

* Pima Indians Diabetes dataset
* Breast Cancer Wisconsin dataset
* Thyroid Disease / Hypothyroid dataset

Dataset rights and citations remain with their original sources.

## Limitations

* This project is an educational machine learning benchmark, not a medical diagnosis system.
* The models are intentionally simple and are used for comparison and learning purposes.
* The datasets are small structured benchmark datasets and may not represent real clinical deployment conditions.
* Model outputs should not be used for medical decision-making.

## License

This project code is licensed under the MIT License.
