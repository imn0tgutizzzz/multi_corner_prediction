# Objective 3 - Machine Learning Models

This branch contains the implementation and evaluation of machine learning models for predicting delay in the **Slow** and **Fast** process corners using information from the **Typical** corner.

## Files

- `modelo_objetivo3.py`: trains and evaluates the machine learning models.
- `graficas_objetivo3.py`: generates comparison plots between the baseline, model predictions, and real delay values.
- `treated_labels_train_typical.csv`
- `treated_labels_train_slow.csv`
- `treated_labels_train_fast.csv`
- `treated_labels_typical.csv`
- `treated_labels_slow.csv`
- `treated_labels_fast.csv`
- `treated_test_designs_typical.csv`
- `treated_test_designs_slow.csv`
- `treated_test_designs_fast.csv`
- `resultados_objetivo3/`: stores metrics, experiments, predictions, trained models, and plots.

## Models

The current implementation evaluates:

- **Baseline**: uses `Delay_Typical` directly as the prediction for Slow or Fast.
- **Ridge Regression**.
- **Random Forest**.
- **Gradient Boosting**.

Random Forest experiments are performed using different numbers of trees:

```text
10
50
100
200
```

The current experiments use:

```text
max_depth = 20
min_samples_leaf = 2
```

The best Random Forest configuration found during validation was:

```text
n_estimators = 200
max_depth = 20
min_samples_leaf = 2
```

The model uses **16 input variables**, including Fanout, Cap, Slew, `Delay_Typical`, coordinates, context variables, and cell sizes.

## Evaluation

The training dataset contains **75,115 observations** and is divided into:

- Training: **60,092 samples**
- Validation: **15,023 samples**

Additional evaluation datasets:

- Test Labels: **23,502 samples**
- Test Designs: **46,971 samples**

The metrics used are:

- MAE
- RMSE
- R²
- Pearson correlation

### Validation Results - Best Random Forest

| Corner | RMSE | R² | Pearson |
| --- | ---: | ---: | ---: |
| Slow | 0.218798 | 0.992713 | 0.996350 |
| Fast | 0.305694 | 0.985403 | 0.992676 |

### Test Designs Results

| Model | Corner | RMSE | R² | Pearson |
| --- | --- | ---: | ---: | ---: |
| Baseline | Slow | 0.334651 | 0.926345 | 0.963778 |
| Random Forest | Slow | 0.329911 | 0.928417 | 0.963958 |
| Baseline | Fast | 0.367934 | 0.885589 | 0.949983 |
| Random Forest | Fast | 0.367309 | 0.885977 | 0.944638 |

## Generated Results

The execution generates:

```text
resultados_objetivo3/
├── metricas_objetivo3.csv
├── experimentos_random_forest.csv
├── predicciones_test_labels.csv
├── predicciones_test_designs.csv
├── modelo_random_forest_slow.joblib
├── modelo_random_forest_fast.joblib
└── graficas/
```

The generated plots compare:

```text
Baseline Typical vs Slow real
Baseline Typical vs Fast real
Random Forest Slow prediction vs Slow real
Random Forest Fast prediction vs Fast real
```

Each comparison plot includes:

- RMSE
- Pearson correlation
- R²

## Usage

Run the machine learning experiments with:

```bash
python modelo_objetivo3.py
```

Generate the comparison plots with:

```bash
python graficas_objetivo3.py
```