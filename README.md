# Objective 3 - Machine Learning Models

This branch contains the implementation and evaluation of machine learning models for predicting delay in the **Slow** and **Fast** process corners using information from the **Typical** corner.

## Files

- `modelo_objetivo3.py`: trains and evaluates the models.
- `treated_labels_train_typical.csv`
- `treated_labels_train_slow.csv`
- `treated_labels_train_fast.csv`
- `treated_test_designs_typical.csv`
- `treated_test_designs_slow.csv`
- `treated_test_designs_fast.csv`
- `resultados_objetivo3/`: stores metrics, predictions, and trained models.

## Models

The current implementation evaluates:

- Baseline
- Ridge Regression
- Random Forest

The input contains 16 variables, including Fanout, Cap, Slew, Delay_Typical, coordinates, context variables, and cell sizes.

## Evaluation

The dataset is divided into:

- Training: 60,092 samples
- Validation: 15,023 samples
- Test Designs: 46,971 samples

Main validation results:

| Model | Corner | RMSE | R² |
|---|---|---:|---:|
| Random Forest | Slow | 0.218818 | 0.992712 |
| Random Forest | Fast | 0.308295 | 0.985154 |

Results on Test Designs:

| Corner | RMSE | R² |
|---|---:|---:|
| Slow | 0.329762 | 0.928481 |
| Fast | 0.367506 | 0.885855 |

## Usage

Run:

```bash
python modelo_objetivo3.py