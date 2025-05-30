# Pet Adoption Prediction

This project implements a deep neural network using **Keras** to predict whether a pet will be adopted based on a dataset of pet characteristics. The goal is to preprocess the dataset, build and train a neural network, and generate binary predictions (`True`/`False`) for a test dataset.

## Project Overview

The project involves the following steps:
1. **Data Loading and Exploration**: Load training and test datasets containing pet attributes.
2. **Data Preprocessing**: Transform the dataset to be suitable for a deep learning model.
3. **Model Building**: Construct a neural network with Keras to predict adoption outcomes.
4. **Model Training**: Train the model on the preprocessed data.
5. **Prediction and Submission**: Generate predictions for the test dataset and save them in the required format.
6. **Evaluation**: Assess the model using the F1 score (weighted average) on a validation set.

## Dataset

The dataset contains information about pets available for adoption. The training dataset includes a column `AdoptionSpeed` (0–4), which is used to create the binary target variable (`Target`):
- **True**: Pet is adopted (AdoptionSpeed 0–3, adopted within 90 days).
- **False**: Pet is not adopted (AdoptionSpeed 4, not adopted after 100 days).

### Features
| Feature          | Description                              |
|------------------|------------------------------------------|
| `Type`           | Type of animal (e.g., dog, cat)          |
| `Age`            | Age in months                            |
| `Breed1`         | Primary breed                            |
| `Gender`         | Gender of the pet                        |
| `Color1`         | Primary color                            |
| `Color2`         | Secondary color (if applicable)          |
| `MaturitySize`   | Size at maturity                         |
| `FurLength`      | Length of fur                            |
| `Vaccinated`     | Vaccination status                       |
| `Sterilized`     | Sterilization status                     |
| `Health`         | Health condition                         |
| `Fee`            | Adoption fee                             |
| `PhotoAmt`       | Number of uploaded photos                |
| `Description`    | Text description (dropped during preprocessing) |
| `AdoptionSpeed`  | Speed of adoption (0–4, in training data only) |

### Target
- `Target`: Binary label indicating whether the pet is adopted (`True`) or not (`False`).

## Preprocessing

The dataset is preprocessed as follows:
1. **Target Creation**: Create a binary `Target` column in the training data (`True` for `AdoptionSpeed` < 4, `False` otherwise).
2. **Drop Columns**: Remove `AdoptionSpeed` and `Description` from the training data, and `Description` from the test data.
3. **Encoding Categorical Features**:
   - **Ordinal Features** (`MaturitySize`, `FurLength`, `Health`): Encoded using `LabelEncoder` from `sklearn`.
   - **Nominal Features** (`Type`, `Gender`): Encoded using `LabelEncoder` (as they have only 2 categories).
   - **Nominal Features with Multiple Categories** (`Breed1`, `Color1`, `Color2`, `Vaccinated`, `Sterilized`): Encoded using `BinaryEncoder` from `category_encoders` to avoid ordinal assumptions.
4. **Normalization**: All features (except `Target`) are normalized using the mean and standard deviation from the training data to achieve a mean of 0 and a standard deviation of 1.
5. **Data Splitting**: Split the training data into 90% training and 10% validation sets using `train_test_split` from `sklearn`.

## Model Architecture

A sequential neural network is built using Keras with the following layers:
- **Input Layer**: Matches the number of features in the preprocessed dataset.
- **Dense Layer 1**: 5000 neurons, ReLU activation.
- **Dense Layer 2**: 1000 neurons, ReLU activation.
- **Dense Layer 3**: 500 neurons, ReLU activation.
- **Output Layer**: 1 neuron, sigmoid activation (for binary classification).

## Training

The model is compiled and trained with:
- **Optimizer**: Adam.
- **Loss Function**: Binary Crossentropy (without logits).
- **Metrics**: Accuracy.
- **Batch Size**: 128.
- **Epochs**: 10.
- **Validation Data**: 10% of the training data.

## Evaluation

The model is evaluated on the validation set using the **F1 score** (weighted average) from `sklearn.metrics.f1_score`. The goal is to achieve an F1 score of at least 0.70.

## Submission

Predictions are generated for the test dataset and saved in a DataFrame named `submission` with a single column `Target` containing binary values (`True`/`False`). The submission DataFrame has 1000 rows, matching the test dataset.

### Submission Format
| Target |
|--------|
| True   |
| False  |
| True   |
| ...    |

The submission is saved as `submission.csv`. Additionally, preprocessed training and test datasets, and the test dataset with predictions, are saved as `train_preprocessed.csv`, `test_preprocessed.csv`, and `test_with_predictions.csv`, respectively.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Keras (with TensorFlow backend)
- scikit-learn
- category_encoders

## How to Run

1. Ensure all dependencies are installed.
2. Place the `petfinder_train.csv` and `petfinder_test.csv` files in the `./data/` directory.
3. Run the provided Python script to preprocess the data, train the model, and generate predictions.
4. The output files (`submission.csv`, `train_preprocessed.csv`, `test_preprocessed.csv`, `test_with_predictions.csv`) will be saved in the current directory.

## Files

- `petfinder_train.csv`: Training dataset.
- `petfinder_test.csv`: Test dataset.
- `train_preprocessed.csv`: Preprocessed training data.
- `test_preprocessed.csv`: Preprocessed test data.
- `submission.csv`: Final predictions for the test dataset.
- `test_with_predictions.csv`: Test data with predicted `Target` values.

## Notes

- The preprocessing steps ensure that the test data is transformed using statistics (mean, standard deviation, encodings) derived from the training data to avoid data leakage.
- The model architecture and training parameters are designed to balance complexity and performance for the binary classification task.