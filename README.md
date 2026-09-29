# Multi-Corner Delay Prediction using Machine Learning

A machine learning implementation for predicting timing delay across different process corners in digital integrated circuits.

The project focuses on using timing information from the **typical process corner** to predict the corresponding delays in the **slow** and **fast** process corners.

**Authors:** María José Guevara Matarrita, Luis Esteban Torres Solís & Jesy Pricilla Rivera Duarte

This project was developed as a graduation project at the **Universidad de Costa Rica**, on behalf of the **Microelectronics and Computer Architecture Research Lab (LIMA)**, under the guidance of **Prof. Erick Carvajal Barboza, Ph.D.**

The project extends previous work on machine learning-based timing prediction by introducing a multi-corner prediction approach. The methodology analyzes the relationship between the Typical, Slow, and Fast process corners and develops machine learning models capable of estimating Slow and Fast delay values using information from the Typical corner.

---

## Project Structure

| File / Branch | Description |
|---|---|
| `main` | Contains the general project documentation and repository overview. |
| `objetivo1` | Contains the original datasets, preprocessing scripts, process-corner separation, and multicorner dataset generation. |
| `objetivo2` | Contains the statistical analysis and visual comparison between Typical, Slow, and Fast delays. |
| `objetivo3` | Contains the machine learning models, experiments, predictions, and evaluation results. |
| `data.py` | Cleans and processes the original CSV files and separates the datasets into Typical, Slow, and Fast process corners. |
| `create_train_multicorner.py` | Matches timing segments between corners and generates the multicorner datasets. |
| `modelo_objetivo3.py` | Trains and evaluates machine learning models for Slow and Fast delay prediction. |

---

## Usage

The project is organized into different Git branches according to each objective.

To work with the dataset preparation stage:

```bash
git checkout objetivo1
```

To work with the statistical analysis stage:

```bash
git checkout objetivo2
```

To work with the machine learning stage:

```bash
git checkout objetivo3
```

## About the Data

The project starts from three original datasets:

```text
data_original/
├── train.csv
├── test_labels.csv
└── test_designs.csv
```

The datasets contain timing information corresponding to different process corners.

The preprocessing stage separates the samples into:

- Typical
- Slow
- Fast

For example:

```text
treated_train_typical.csv
treated_train_slow.csv
treated_train_fast.csv
```

Equivalent files are generated for `test_labels` and `test_designs`.

The correspondence between samples from different corners is established using:

- `Description`
- `Previous_description`

These fields identify the same timing segment across the different process corners.

After matching the corresponding samples, the following multicorner datasets are generated:

```text
train_multicorner.csv
test_labels_multicorner.csv
test_designs_multicorner.csv
```

The **Typical** corner is used as the main reference, while the corresponding **Slow** and **Fast** delay values are associated with each timing segment.

## Data Preprocessing

The `data.py` script is responsible for:

- Cleaning the original CSV files.
- Removing unnecessary information.
- Preparing the datasets for further analysis.
- Separating the data into Typical, Slow, and Fast process corners.

The resulting files are then processed by:

```bash
python create_train_multicorner.py
```

This script matches the corresponding timing segments between corners and generates the final multicorner datasets.

The general preprocessing flow is:

```text
Original datasets
        ↓
      data.py
        ↓
Typical / Slow / Fast datasets
        ↓
create_train_multicorner.py
        ↓
Multicorner datasets
```

## Process Corner Analysis

Before training the machine learning models, the relationship between the different process corners is analyzed.

The following comparisons are performed:

- Typical vs Slow
- Typical vs Fast

For the following datasets:

- Train
- Test Labels
- Test Designs

Scatter plots are generated to visualize the relationship between the delay values.

The **Pearson correlation coefficient** is also calculated to quantify the correlation between the different process corners.

This analysis serves as a sanity check to verify that information from the Typical corner can be useful for estimating the Slow and Fast corners.

## About the Machine Learning Model

The prediction problem is divided into two regression tasks:

```text
Typical corner information → Slow delay
Typical corner information → Fast delay
```

The Typical-corner information is used as the model input, while the corresponding Slow or Fast delay is used as the prediction target.

The initial implementation uses **Random Forest regression**.

Different model configurations can be evaluated by modifying parameters such as:

- `n_estimators`
- `max_depth`

Other machine learning models, including **Gradient Boosted Decision Trees (GBDT)** or **XGBoost**, may also be evaluated.

## Model Evaluation

The machine learning models are evaluated using:

- RMSE
- Pearson correlation
- R²

Results are evaluated separately for:

- Training
- Validation
- Test Labels
- Test Designs

For each experiment, the model configuration and its corresponding metrics should be documented in order to compare different approaches.

The results generated by the current implementation are stored in:

```text
resultados_objetivo3/
├── metricas_objetivo3.csv
└── predicciones_test_designs.csv
```

Additional experiment results can be added as new model configurations are evaluated.