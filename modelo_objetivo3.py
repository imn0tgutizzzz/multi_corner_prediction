from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR

RESULTS_DIR = BASE_DIR / "resultados_objetivo3"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# ARCHIVOS
# ============================================================

TRAIN_TYPICAL = DATA_DIR / "treated_labels_train_typical.csv"
TRAIN_SLOW = DATA_DIR / "treated_labels_train_slow.csv"
TRAIN_FAST = DATA_DIR / "treated_labels_train_fast.csv"

TEST_TYPICAL = DATA_DIR / "treated_test_designs_typical.csv"
TEST_SLOW = DATA_DIR / "treated_test_designs_slow.csv"
TEST_FAST = DATA_DIR / "treated_test_designs_fast.csv"


# ============================================================
# FUNCIONES
# ============================================================

def calcular_metricas(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_real, y_pred)
    )

    r2 = r2_score(y_real, y_pred)

    return mae, rmse, r2


def evaluar_modelo(nombre, y_real, y_pred, conjunto, esquina):
    mae, rmse, r2 = calcular_metricas(y_real, y_pred)

    return {
        "Modelo": nombre,
        "Conjunto": conjunto,
        "Esquina": esquina,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# VERIFICAR ARCHIVOS
# ============================================================

archivos = [
    TRAIN_TYPICAL,
    TRAIN_SLOW,
    TRAIN_FAST,
    TEST_TYPICAL,
    TEST_SLOW,
    TEST_FAST,
]

print("=" * 70)
print("VERIFICACION DE ARCHIVOS")
print("=" * 70)

for archivo in archivos:
    if not archivo.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo:\n{archivo}"
        )

    print(f"OK: {archivo.name}")


# ============================================================
# CARGAR DATOS DE ENTRENAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("CARGANDO DATOS DE ENTRENAMIENTO")
print("=" * 70)

train_typical = pd.read_csv(TRAIN_TYPICAL)
train_slow = pd.read_csv(TRAIN_SLOW)
train_fast = pd.read_csv(TRAIN_FAST)

print(f"Typical: {train_typical.shape}")
print(f"Slow:    {train_slow.shape}")
print(f"Fast:    {train_fast.shape}")


# ============================================================
# VERIFICAR CORRESPONDENCIA
# ============================================================

print("\n" + "=" * 70)
print("VERIFICANDO CORRESPONDENCIA ENTRE ESQUINAS")
print("=" * 70)

columnas_llave = [
    "Previous_description",
    "Description",
]

# verificar duplicados de la llave
if train_typical.duplicated(columnas_llave).any():
    raise ValueError(
        "Hay segmentos duplicados en Typical."
    )

if train_slow.duplicated(columnas_llave).any():
    raise ValueError(
        "Hay segmentos duplicados en Slow."
    )

if train_fast.duplicated(columnas_llave).any():
    raise ValueError(
        "Hay segmentos duplicados en Fast."
    )

# crear las llaves
llaves_typical = set(
    zip(
        train_typical["Previous_description"],
        train_typical["Description"],
    )
)

llaves_slow = set(
    zip(
        train_slow["Previous_description"],
        train_slow["Description"],
    )
)

llaves_fast = set(
    zip(
        train_fast["Previous_description"],
        train_fast["Description"],
    )
)

# comprobar que las tres esquinas contienen los mismos segmentos
if llaves_typical != llaves_slow:
    raise ValueError(
        "Typical y Slow no contienen los mismos segmentos."
    )

if llaves_typical != llaves_fast:
    raise ValueError(
        "Typical y Fast no contienen los mismos segmentos."
    )

print(
    "Los tres archivos contienen los mismos segmentos."
)

print(
    f"Segmentos correspondientes: {len(llaves_typical)}"
)


# ============================================================
# CREAR TABLA DE ENTRENAMIENTO
# ============================================================

print("\n" + "=" * 70)
print("CREANDO DATASET MULTICORNER PARA MACHINE LEARNING")
print("=" * 70)

# seleccionar solamente la llave y el Delay de Slow
slow_delay = train_slow[
    columnas_llave + ["Delay"]
].rename(
    columns={
        "Delay": "Delay_Slow"
    }
)

# seleccionar solamente la llave y el Delay de Fast
fast_delay = train_fast[
    columnas_llave + ["Delay"]
].rename(
    columns={
        "Delay": "Delay_Fast"
    }
)

# Typical es la base de las variables de entrada
datos_train = train_typical.rename(
    columns={
        "Delay": "Delay_Typical"
    }
).copy()

# unir Slow utilizando la llave correcta
datos_train = datos_train.merge(
    slow_delay,
    on=columnas_llave,
    how="inner",
    validate="one_to_one",
)

# unir Fast utilizando la llave correcta
datos_train = datos_train.merge(
    fast_delay,
    on=columnas_llave,
    how="inner",
    validate="one_to_one",
)

print(
    f"Dataset final de entrenamiento: {datos_train.shape}"
)

if len(datos_train) != len(train_typical):
    raise ValueError(
        "Se perdieron filas al unir Typical, Slow y Fast."
    )

print(
    "La union Typical-Slow-Fast fue correcta."
)


# ============================================================
# VARIABLES DE ENTRADA
# ============================================================

columnas_excluir = [
    "row_id",
    "Previous_description",
    "Description",
    "Delta",
    "Delay_Slow",
    "Delay_Fast",
]

columnas_features = [
    columna
    for columna in datos_train.columns
    if columna not in columnas_excluir
]

print("\nVariables utilizadas por el modelo:")

for columna in columnas_features:
    print(f"  - {columna}")


# ============================================================
# MATRIZ X Y OBJETIVOS
# ============================================================

X = datos_train[columnas_features].copy()

y_slow = datos_train["Delay_Slow"].copy()
y_fast = datos_train["Delay_Fast"].copy()


# ============================================================
# ASEGURAR QUE LAS VARIABLES SEAN NUMERICAS
# ============================================================

X = X.apply(pd.to_numeric, errors="coerce")

print("\nCantidad de variables de entrada:", X.shape[1])
print("Cantidad de observaciones:", X.shape[0])


# ============================================================
# DIVISION TRAIN / VALIDACION
# ============================================================

indices = np.arange(len(X))

indices_train, indices_validacion = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
)

X_train = X.iloc[indices_train]
X_validacion = X.iloc[indices_validacion]

y_slow_train = y_slow.iloc[indices_train]
y_slow_validacion = y_slow.iloc[indices_validacion]

y_fast_train = y_fast.iloc[indices_train]
y_fast_validacion = y_fast.iloc[indices_validacion]


print("\n" + "=" * 70)
print("DIVISION DE DATOS")
print("=" * 70)

print(f"Entrenamiento: {len(X_train)}")
print(f"Validacion:    {len(X_validacion)}")


# ============================================================
# BASELINE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)

baseline_slow = datos_train.iloc[
    indices_validacion
]["Delay_Typical"].values

baseline_fast = datos_train.iloc[
    indices_validacion
]["Delay_Typical"].values

resultados = []

resultados.append(
    evaluar_modelo(
        "Baseline",
        y_slow_validacion,
        baseline_slow,
        "Validacion",
        "Slow",
    )
)

resultados.append(
    evaluar_modelo(
        "Baseline",
        y_fast_validacion,
        baseline_fast,
        "Validacion",
        "Fast",
    )
)


# ============================================================
# MODELO RIDGE - SLOW
# ============================================================

print("\n" + "=" * 70)
print("ENTRENANDO RIDGE PARA SLOW")
print("=" * 70)

modelo_ridge_slow = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "modelo",
            Ridge(alpha=1.0),
        ),
    ]
)

modelo_ridge_slow.fit(
    X_train,
    y_slow_train,
)

pred_ridge_slow = modelo_ridge_slow.predict(
    X_validacion
)

resultados.append(
    evaluar_modelo(
        "Ridge",
        y_slow_validacion,
        pred_ridge_slow,
        "Validacion",
        "Slow",
    )
)


# ============================================================
# MODELO RIDGE - FAST
# ============================================================

print("\n" + "=" * 70)
print("ENTRENANDO RIDGE PARA FAST")
print("=" * 70)

modelo_ridge_fast = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "modelo",
            Ridge(alpha=1.0),
        ),
    ]
)

modelo_ridge_fast.fit(
    X_train,
    y_fast_train,
)

pred_ridge_fast = modelo_ridge_fast.predict(
    X_validacion
)

resultados.append(
    evaluar_modelo(
        "Ridge",
        y_fast_validacion,
        pred_ridge_fast,
        "Validacion",
        "Fast",
    )
)


# ============================================================
# RANDOM FOREST - SLOW
# ============================================================

print("\n" + "=" * 70)
print("ENTRENANDO RANDOM FOREST PARA SLOW")
print("=" * 70)

modelo_rf_slow = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "modelo",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

modelo_rf_slow.fit(
    X_train,
    y_slow_train,
)

pred_rf_slow = modelo_rf_slow.predict(
    X_validacion
)

resultados.append(
    evaluar_modelo(
        "Random Forest",
        y_slow_validacion,
        pred_rf_slow,
        "Validacion",
        "Slow",
    )
)


# ============================================================
# RANDOM FOREST - FAST
# ============================================================

print("\n" + "=" * 70)
print("ENTRENANDO RANDOM FOREST PARA FAST")
print("=" * 70)

modelo_rf_fast = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "modelo",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

modelo_rf_fast.fit(
    X_train,
    y_fast_train,
)

pred_rf_fast = modelo_rf_fast.predict(
    X_validacion
)

resultados.append(
    evaluar_modelo(
        "Random Forest",
        y_fast_validacion,
        pred_rf_fast,
        "Validacion",
        "Fast",
    )
)


# ============================================================
# RESULTADOS DE VALIDACION
# ============================================================

resultados_df = pd.DataFrame(resultados)

print("\n" + "=" * 70)
print("RESULTADOS DE VALIDACION")
print("=" * 70)

print(
    resultados_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# CARGAR TEST DESIGNS
# ============================================================

print("\n" + "=" * 70)
print("CARGANDO TEST DESIGNS")
print("=" * 70)

test_typical = pd.read_csv(TEST_TYPICAL)
test_slow = pd.read_csv(TEST_SLOW)
test_fast = pd.read_csv(TEST_FAST)

print(f"Typical: {test_typical.shape}")
print(f"Slow:    {test_slow.shape}")
print(f"Fast:    {test_fast.shape}")


# ============================================================
# VERIFICAR CORRESPONDENCIA DE TEST DESIGNS
# ============================================================

print("\n" + "=" * 70)
print("VERIFICANDO CORRESPONDENCIA DE TEST DESIGNS")
print("=" * 70)

# verificar duplicados
if test_typical.duplicated(columnas_llave).any():
    raise ValueError(
        "Hay segmentos duplicados en test Typical."
    )

if test_slow.duplicated(columnas_llave).any():
    raise ValueError(
        "Hay segmentos duplicados en test Slow."
    )

if test_fast.duplicated(columnas_llave).any():
    raise ValueError(
        "Hay segmentos duplicados en test Fast."
    )

# crear las llaves
llaves_test_typical = set(
    zip(
        test_typical["Previous_description"],
        test_typical["Description"],
    )
)

llaves_test_slow = set(
    zip(
        test_slow["Previous_description"],
        test_slow["Description"],
    )
)

llaves_test_fast = set(
    zip(
        test_fast["Previous_description"],
        test_fast["Description"],
    )
)

if llaves_test_typical != llaves_test_slow:
    raise ValueError(
        "Test Typical y Slow no contienen los mismos segmentos."
    )

if llaves_test_typical != llaves_test_fast:
    raise ValueError(
        "Test Typical y Fast no contienen los mismos segmentos."
    )

print(
    "Los tres test designs contienen los mismos segmentos."
)

print(
    f"Segmentos correspondientes: {len(llaves_test_typical)}"
)


# ============================================================
# CREAR DATASET MULTICORNER DE TEST
# ============================================================

test_slow_delay = test_slow[
    columnas_llave + ["Delay"]
].rename(
    columns={
        "Delay": "Delay_Slow"
    }
)

test_fast_delay = test_fast[
    columnas_llave + ["Delay"]
].rename(
    columns={
        "Delay": "Delay_Fast"
    }
)

test_datos = test_typical.rename(
    columns={
        "Delay": "Delay_Typical"
    }
).copy()

# unir Slow
test_datos = test_datos.merge(
    test_slow_delay,
    on=columnas_llave,
    how="inner",
    validate="one_to_one",
)

# unir Fast
test_datos = test_datos.merge(
    test_fast_delay,
    on=columnas_llave,
    how="inner",
    validate="one_to_one",
)

print(
    f"Dataset final de test: {test_datos.shape}"
)

if len(test_datos) != len(test_typical):
    raise ValueError(
        "Se perdieron filas al unir los test designs."
    )

print(
    "La union de los test designs fue correcta."
)


# ============================================================
# PREPARAR VARIABLES DE TEST
# ============================================================

X_test = test_datos[columnas_features].copy()

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce",
)

y_test_slow = test_datos["Delay_Slow"]

y_test_fast = test_datos["Delay_Fast"]

# ============================================================
# ENTRENAR MODELOS FINALES CON TODO EL TRAIN
# ============================================================

print("\n" + "=" * 70)
print("ENTRENANDO MODELOS FINALES CON TODO EL DATASET")
print("=" * 70)

modelo_final_slow = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "modelo",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

modelo_final_fast = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "modelo",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

modelo_final_slow.fit(
    X,
    y_slow,
)

modelo_final_fast.fit(
    X,
    y_fast,
)


# ============================================================
# PREDICCIONES SOBRE TEST DESIGNS
# ============================================================

print("\n" + "=" * 70)
print("EVALUANDO EN TEST DESIGNS")
print("=" * 70)

pred_test_slow = modelo_final_slow.predict(X_test)
pred_test_fast = modelo_final_fast.predict(X_test)


# ============================================================
# METRICAS TEST
# ============================================================

resultado_test_slow = evaluar_modelo(
    "Random Forest",
    y_test_slow,
    pred_test_slow,
    "Test Designs",
    "Slow",
)

resultado_test_fast = evaluar_modelo(
    "Random Forest",
    y_test_fast,
    pred_test_fast,
    "Test Designs",
    "Fast",
)

resultados_test = pd.DataFrame(
    [
        resultado_test_slow,
        resultado_test_fast,
    ]
)

print(
    resultados_test.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

resultados_completos = pd.concat(
    [
        resultados_df,
        resultados_test,
    ],
    ignore_index=True,
)

ruta_resultados = RESULTS_DIR / "metricas_objetivo3.csv"

resultados_completos.to_csv(
    ruta_resultados,
    index=False,
)


# ============================================================
# GUARDAR PREDICCIONES
# ============================================================

predicciones = test_datos[
    [
        "row_id",
        "Delay_Typical",
        "Delay_Slow",
        "Delay_Fast",
    ]
].copy()

predicciones["Predicted_Slow"] = pred_test_slow
predicciones["Predicted_Fast"] = pred_test_fast

ruta_predicciones = RESULTS_DIR / "predicciones_test_designs.csv"

predicciones.to_csv(
    ruta_predicciones,
    index=False,
)


# ============================================================
# GUARDAR MODELOS
# ============================================================

import joblib

joblib.dump(
    modelo_final_slow,
    RESULTS_DIR / "modelo_random_forest_slow.joblib",
)

joblib.dump(
    modelo_final_fast,
    RESULTS_DIR / "modelo_random_forest_fast.joblib",
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("PROCESO TERMINADO")
print("=" * 70)

print(f"Resultados:")
print(ruta_resultados)

print(f"\nPredicciones:")
print(ruta_predicciones)

print("\nModelos guardados en:")
print(RESULTS_DIR)

print("\nObjetivo 3 ejecutado correctamente.")