from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

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

GRAFICAS_DIR = RESULTS_DIR / "graficas"

GRAFICAS_DIR.mkdir(exist_ok=True)


# ============================================================
# ARCHIVOS DE ENTRENAMIENTO
# ============================================================

TRAIN_TYPICAL = DATA_DIR / "treated_labels_train_typical.csv"
TRAIN_SLOW = DATA_DIR / "treated_labels_train_slow.csv"
TRAIN_FAST = DATA_DIR / "treated_labels_train_fast.csv"


# ============================================================
# ARCHIVOS TEST LABELS
# ============================================================

TEST_LABELS_TYPICAL = DATA_DIR / "treated_labels_typical.csv"
TEST_LABELS_SLOW = DATA_DIR / "treated_labels_slow.csv"
TEST_LABELS_FAST = DATA_DIR / "treated_labels_fast.csv"


# ============================================================
# ARCHIVOS TEST DESIGNS
# ============================================================

TEST_DESIGNS_TYPICAL = DATA_DIR / "treated_test_designs_typical.csv"
TEST_DESIGNS_SLOW = DATA_DIR / "treated_test_designs_slow.csv"
TEST_DESIGNS_FAST = DATA_DIR / "treated_test_designs_fast.csv"


# ============================================================
# FUNCIONES GENERALES
# ============================================================

def calcular_metricas(y_real, y_pred):

    mae = mean_absolute_error(
        y_real,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_real,
            y_pred,
        )
    )

    r2 = r2_score(
        y_real,
        y_pred,
    )

    pearson = np.corrcoef(
        y_real,
        y_pred,
    )[0, 1]

    return mae, rmse, r2, pearson


# ============================================================
# GUARDAR RESULTADO
# ============================================================

def crear_resultado(
    modelo,
    experimento,
    conjunto,
    esquina,
    y_real,
    y_pred,
):

    mae, rmse, r2, pearson = calcular_metricas(
        y_real,
        y_pred,
    )

    return {
        "Modelo": modelo,
        "Experimento": experimento,
        "Conjunto": conjunto,
        "Esquina": esquina,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Pearson": pearson,
    }


# ============================================================
# VERIFICAR ARCHIVOS
# ============================================================

archivos = [

    TRAIN_TYPICAL,
    TRAIN_SLOW,
    TRAIN_FAST,

    TEST_LABELS_TYPICAL,
    TEST_LABELS_SLOW,
    TEST_LABELS_FAST,

    TEST_DESIGNS_TYPICAL,
    TEST_DESIGNS_SLOW,
    TEST_DESIGNS_FAST,
]


print("=" * 70)
print("VERIFICACION DE ARCHIVOS")
print("=" * 70)


for archivo in archivos:

    if not archivo.exists():

        raise FileNotFoundError(
            f"No se encontro el archivo:\n{archivo}"
        )

    print(
        f"OK: {archivo.name}"
    )


# ============================================================
# CARGAR TRAIN
# ============================================================

print("\n" + "=" * 70)
print("CARGANDO DATOS DE ENTRENAMIENTO")
print("=" * 70)


train_typical = pd.read_csv(
    TRAIN_TYPICAL
)

train_slow = pd.read_csv(
    TRAIN_SLOW
)

train_fast = pd.read_csv(
    TRAIN_FAST
)


print(
    f"Typical: {train_typical.shape}"
)

print(
    f"Slow:    {train_slow.shape}"
)

print(
    f"Fast:    {train_fast.shape}"
)


# ============================================================
# LLAVE PARA IDENTIFICAR CADA CAMINO
# ============================================================

columnas_llave = [
    "Previous_description",
    "Description",
]


# ============================================================
# VERIFICAR CORRESPONDENCIA
# ============================================================

print("\n" + "=" * 70)
print("VERIFICANDO CORRESPONDENCIA ENTRE ESQUINAS")
print("=" * 70)


for nombre, datos in [

    ("Typical", train_typical),
    ("Slow", train_slow),
    ("Fast", train_fast),

]:

    if datos.duplicated(
        columnas_llave
    ).any():

        raise ValueError(
            f"Hay caminos duplicados en {nombre}."
        )


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


if llaves_typical != llaves_slow:

    raise ValueError(
        "Typical y Slow no contienen los mismos caminos."
    )


if llaves_typical != llaves_fast:

    raise ValueError(
        "Typical y Fast no contienen los mismos caminos."
    )


print(
    "Los tres archivos contienen los mismos segmentos."
)


print(
    f"Segmentos correspondientes: "
    f"{len(llaves_typical):,}"
)


# ============================================================
# CREAR DATASET MULTICORNER
# ============================================================

print("\n" + "=" * 70)
print("CREANDO DATASET MULTICORNER")
print("=" * 70)


slow_delay = train_slow[
    columnas_llave + ["Delay"]
].rename(
    columns={
        "Delay": "Delay_Slow"
    }
)


fast_delay = train_fast[
    columnas_llave + ["Delay"]
].rename(
    columns={
        "Delay": "Delay_Fast"
    }
)


datos_train = train_typical.rename(
    columns={
        "Delay": "Delay_Typical"
    }
).copy()


datos_train = datos_train.merge(
    slow_delay,
    on=columnas_llave,
    how="inner",
    validate="one_to_one",
)


datos_train = datos_train.merge(
    fast_delay,
    on=columnas_llave,
    how="inner",
    validate="one_to_one",
)


print(
    f"Dataset final: {datos_train.shape}"
)


if len(datos_train) != len(train_typical):

    raise ValueError(
        "Se perdieron caminos durante la union."
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

    print(
        f"  - {columna}"
    )


print(
    f"\nCantidad de variables: "
    f"{len(columnas_features)}"
)


print(
    f"Cantidad de observaciones: "
    f"{len(datos_train):,}"
)


# ============================================================
# MATRIZ X Y VARIABLES OBJETIVO
# ============================================================

X = datos_train[
    columnas_features
].copy()


X = X.apply(
    pd.to_numeric,
    errors="coerce",
)


y_slow = datos_train[
    "Delay_Slow"
].copy()


y_fast = datos_train[
    "Delay_Fast"
].copy()


# ============================================================
# DIVISION TRAIN / VALIDACION
# ============================================================

print("\n" + "=" * 70)
print("DIVISION DE DATOS")
print("=" * 70)


indices = np.arange(
    len(X)
)


indices_train, indices_validacion = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
)


X_train = X.iloc[
    indices_train
].copy()


X_validacion = X.iloc[
    indices_validacion
].copy()


y_slow_train = y_slow.iloc[
    indices_train
].copy()


y_slow_validacion = y_slow.iloc[
    indices_validacion
].copy()


y_fast_train = y_fast.iloc[
    indices_train
].copy()


y_fast_validacion = y_fast.iloc[
    indices_validacion
].copy()


print(
    f"Entrenamiento: {len(X_train):,}"
)


print(
    f"Validacion:    {len(X_validacion):,}"
)


# ============================================================
# LISTA DE RESULTADOS
# ============================================================

resultados = []


# ============================================================
# PREDICCIONES DEL BASELINE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)


print(
    "Baseline: usar Delay_Typical "
    "como prediccion de Slow y Fast."
)


# ------------------------------
# TRAIN
# ------------------------------

baseline_train_slow = datos_train.iloc[
    indices_train
]["Delay_Typical"].values


baseline_train_fast = datos_train.iloc[
    indices_train
]["Delay_Typical"].values


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Train",
        "Slow",
        y_slow_train,
        baseline_train_slow,
    )
)


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Train",
        "Fast",
        y_fast_train,
        baseline_train_fast,
    )
)


# ------------------------------
# VALIDACION
# ------------------------------

baseline_validacion_slow = datos_train.iloc[
    indices_validacion
]["Delay_Typical"].values


baseline_validacion_fast = datos_train.iloc[
    indices_validacion
]["Delay_Typical"].values


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Validacion",
        "Slow",
        y_slow_validacion,
        baseline_validacion_slow,
    )
)


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Validacion",
        "Fast",
        y_fast_validacion,
        baseline_validacion_fast,
    )
)


# ============================================================
# MODELO RIDGE
# ============================================================

print("\n" + "=" * 70)
print("RIDGE")
print("=" * 70)


modelo_ridge_slow = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "scaler",
            StandardScaler(),
        ),

        (
            "modelo",
            Ridge(
                alpha=1.0
            ),
        ),

    ]
)


modelo_ridge_fast = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "scaler",
            StandardScaler(),
        ),

        (
            "modelo",
            Ridge(
                alpha=1.0
            ),
        ),

    ]
)


# ------------------------------
# RIDGE SLOW
# ------------------------------

modelo_ridge_slow.fit(
    X_train,
    y_slow_train,
)


pred_ridge_slow_train = (
    modelo_ridge_slow.predict(
        X_train
    )
)


pred_ridge_slow_validacion = (
    modelo_ridge_slow.predict(
        X_validacion
    )
)


resultados.append(
    crear_resultado(
        "Ridge",
        "alpha=1.0",
        "Train",
        "Slow",
        y_slow_train,
        pred_ridge_slow_train,
    )
)


resultados.append(
    crear_resultado(
        "Ridge",
        "alpha=1.0",
        "Validacion",
        "Slow",
        y_slow_validacion,
        pred_ridge_slow_validacion,
    )
)


# ------------------------------
# RIDGE FAST
# ------------------------------

modelo_ridge_fast.fit(
    X_train,
    y_fast_train,
)


pred_ridge_fast_train = (
    modelo_ridge_fast.predict(
        X_train
    )
)


pred_ridge_fast_validacion = (
    modelo_ridge_fast.predict(
        X_validacion
    )
)


resultados.append(
    crear_resultado(
        "Ridge",
        "alpha=1.0",
        "Train",
        "Fast",
        y_fast_train,
        pred_ridge_fast_train,
    )
)


resultados.append(
    crear_resultado(
        "Ridge",
        "alpha=1.0",
        "Validacion",
        "Fast",
        y_fast_validacion,
        pred_ridge_fast_validacion,
    )
)


# ============================================================
# EXPERIMENTOS RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENTOS RANDOM FOREST")
print("=" * 70)


experimentos_rf = [

    {
        "trees": 10,
        "depth": 20,
        "leaf": 2,
    },

    {
        "trees": 50,
        "depth": 20,
        "leaf": 2,
    },

    {
        "trees": 100,
        "depth": 20,
        "leaf": 2,
    },

    {
        "trees": 200,
        "depth": 20,
        "leaf": 2,
    },

]


resultados_experimentos_rf = []


modelos_rf_validacion = {}


for configuracion in experimentos_rf:

    trees = configuracion["trees"]

    depth = configuracion["depth"]

    leaf = configuracion["leaf"]


    nombre_experimento = (
        f"trees={trees}_depth={depth}_leaf={leaf}"
    )


    print("\n" + "-" * 70)

    print(
        f"Random Forest: "
        f"{nombre_experimento}"
    )

    print("-" * 70)


    # ========================================================
    # RANDOM FOREST SLOW
    # ========================================================

    modelo_rf_slow = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "modelo",
                RandomForestRegressor(

                    n_estimators=trees,

                    max_depth=depth,

                    min_samples_leaf=leaf,

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


    pred_rf_slow_train = (
        modelo_rf_slow.predict(
            X_train
        )
    )


    pred_rf_slow_validacion = (
        modelo_rf_slow.predict(
            X_validacion
        )
    )


    resultados.append(
        crear_resultado(
            "Random Forest",
            nombre_experimento,
            "Train",
            "Slow",
            y_slow_train,
            pred_rf_slow_train,
        )
    )


    resultado_rf_slow_validacion = crear_resultado(
        "Random Forest",
        nombre_experimento,
        "Validacion",
        "Slow",
        y_slow_validacion,
        pred_rf_slow_validacion,
    )


    resultados.append(
        resultado_rf_slow_validacion
    )


    # ========================================================
    # RANDOM FOREST FAST
    # ========================================================

    modelo_rf_fast = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "modelo",
                RandomForestRegressor(

                    n_estimators=trees,

                    max_depth=depth,

                    min_samples_leaf=leaf,

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


    pred_rf_fast_train = (
        modelo_rf_fast.predict(
            X_train
        )
    )


    pred_rf_fast_validacion = (
        modelo_rf_fast.predict(
            X_validacion
        )
    )


    resultados.append(
        crear_resultado(
            "Random Forest",
            nombre_experimento,
            "Train",
            "Fast",
            y_fast_train,
            pred_rf_fast_train,
        )
    )


    resultado_rf_fast_validacion = crear_resultado(
        "Random Forest",
        nombre_experimento,
        "Validacion",
        "Fast",
        y_fast_validacion,
        pred_rf_fast_validacion,
    )


    resultados.append(
        resultado_rf_fast_validacion
    )


    # guardar informacion del experimento
    resultados_experimentos_rf.append(
        resultado_rf_slow_validacion
    )

    resultados_experimentos_rf.append(
        resultado_rf_fast_validacion
    )


    modelos_rf_validacion[
        nombre_experimento
    ] = {

        "slow": modelo_rf_slow,

        "fast": modelo_rf_fast,

        "configuracion": configuracion,

        "pred_slow": pred_rf_slow_validacion,

        "pred_fast": pred_rf_fast_validacion,

    }


# ============================================================
# GRADIENT BOOSTING
# ============================================================

print("\n" + "=" * 70)
print("GRADIENT BOOSTING")
print("=" * 70)


gb_experimento = (
    "trees=100_learning_rate=0.05_depth=3"
)


modelo_gb_slow = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "modelo",
            GradientBoostingRegressor(

                n_estimators=100,

                learning_rate=0.05,

                max_depth=3,

                random_state=42,

            ),
        ),

    ]
)


modelo_gb_fast = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "modelo",
            GradientBoostingRegressor(

                n_estimators=100,

                learning_rate=0.05,

                max_depth=3,

                random_state=42,

            ),
        ),

    ]
)


# ------------------------------
# GB SLOW
# ------------------------------

modelo_gb_slow.fit(
    X_train,
    y_slow_train,
)


pred_gb_slow_train = (
    modelo_gb_slow.predict(
        X_train
    )
)


pred_gb_slow_validacion = (
    modelo_gb_slow.predict(
        X_validacion
    )
)


resultados.append(
    crear_resultado(
        "Gradient Boosting",
        gb_experimento,
        "Train",
        "Slow",
        y_slow_train,
        pred_gb_slow_train,
    )
)


resultados.append(
    crear_resultado(
        "Gradient Boosting",
        gb_experimento,
        "Validacion",
        "Slow",
        y_slow_validacion,
        pred_gb_slow_validacion,
    )
)


# ------------------------------
# GB FAST
# ------------------------------

modelo_gb_fast.fit(
    X_train,
    y_fast_train,
)


pred_gb_fast_train = (
    modelo_gb_fast.predict(
        X_train
    )
)


pred_gb_fast_validacion = (
    modelo_gb_fast.predict(
        X_validacion
    )
)


resultados.append(
    crear_resultado(
        "Gradient Boosting",
        gb_experimento,
        "Train",
        "Fast",
        y_fast_train,
        pred_gb_fast_train,
    )
)


resultados.append(
    crear_resultado(
        "Gradient Boosting",
        gb_experimento,
        "Validacion",
        "Fast",
        y_fast_validacion,
        pred_gb_fast_validacion,
    )
)


# ============================================================
# RESULTADOS DE VALIDACION
# ============================================================

resultados_df = pd.DataFrame(
    resultados
)


print("\n" + "=" * 70)
print("RESULTADOS DE VALIDACION")
print("=" * 70)


resultados_validacion = resultados_df[
    resultados_df["Conjunto"] == "Validacion"
].copy()


resultados_validacion = resultados_validacion.sort_values(
    [
        "Esquina",
        "RMSE",
    ]
)


print(
    resultados_validacion.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# MEJOR RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("MEJORES CONFIGURACIONES RANDOM FOREST")
print("=" * 70)


rf_validacion = pd.DataFrame(
    resultados_experimentos_rf
)


rf_slow = rf_validacion[
    rf_validacion["Esquina"] == "Slow"
]


rf_fast = rf_validacion[
    rf_validacion["Esquina"] == "Fast"
]


mejor_slow = rf_slow.loc[
    rf_slow["RMSE"].idxmin()
]


mejor_fast = rf_fast.loc[
    rf_fast["RMSE"].idxmin()
]


print("\nSlow:")

print(
    f"Experimento: "
    f"{mejor_slow['Experimento']}"
)

print(
    f"RMSE: "
    f"{mejor_slow['RMSE']:.6f}"
)

print(
    f"R2: "
    f"{mejor_slow['R2']:.6f}"
)

print(
    f"Pearson: "
    f"{mejor_slow['Pearson']:.6f}"
)


print("\nFast:")

print(
    f"Experimento: "
    f"{mejor_fast['Experimento']}"
)

print(
    f"RMSE: "
    f"{mejor_fast['RMSE']:.6f}"
)

print(
    f"R2: "
    f"{mejor_fast['R2']:.6f}"
)

print(
    f"Pearson: "
    f"{mejor_fast['Pearson']:.6f}"
)


mejor_config_slow = next(
    configuracion
    for configuracion in experimentos_rf
    if (
        f"trees={configuracion['trees']}"
        f"_depth={configuracion['depth']}"
        f"_leaf={configuracion['leaf']}"
    )
    == mejor_slow["Experimento"]
)


mejor_config_fast = next(
    configuracion
    for configuracion in experimentos_rf
    if (
        f"trees={configuracion['trees']}"
        f"_depth={configuracion['depth']}"
        f"_leaf={configuracion['leaf']}"
    )
    == mejor_fast["Experimento"]
)


# ============================================================
# CARGAR TEST LABELS
# ============================================================

print("\n" + "=" * 70)
print("CARGANDO TEST LABELS")
print("=" * 70)


test_labels_typical = pd.read_csv(
    TEST_LABELS_TYPICAL
)


test_labels_slow = pd.read_csv(
    TEST_LABELS_SLOW
)


test_labels_fast = pd.read_csv(
    TEST_LABELS_FAST
)


print(
    f"Typical: {test_labels_typical.shape}"
)


print(
    f"Slow:    {test_labels_slow.shape}"
)


print(
    f"Fast:    {test_labels_fast.shape}"
)


# ============================================================
# FUNCION PARA CREAR DATASET MULTICORNER
# ============================================================

def crear_dataset_multicorner(
    datos_typical,
    datos_slow,
    datos_fast,
):

    slow_delay = datos_slow[
        columnas_llave + ["Delay"]
    ].rename(
        columns={
            "Delay": "Delay_Slow"
        }
    )


    fast_delay = datos_fast[
        columnas_llave + ["Delay"]
    ].rename(
        columns={
            "Delay": "Delay_Fast"
        }
    )


    datos = datos_typical.rename(
        columns={
            "Delay": "Delay_Typical"
        }
    ).copy()


    datos = datos.merge(
        slow_delay,
        on=columnas_llave,
        how="inner",
        validate="one_to_one",
    )


    datos = datos.merge(
        fast_delay,
        on=columnas_llave,
        how="inner",
        validate="one_to_one",
    )


    return datos


# ============================================================
# CREAR TEST LABELS
# ============================================================

test_labels = crear_dataset_multicorner(
    test_labels_typical,
    test_labels_slow,
    test_labels_fast,
)


print(
    f"Test Labels final: "
    f"{test_labels.shape}"
)


# ============================================================
# CARGAR TEST DESIGNS
# ============================================================

print("\n" + "=" * 70)
print("CARGANDO TEST DESIGNS")
print("=" * 70)


test_designs_typical = pd.read_csv(
    TEST_DESIGNS_TYPICAL
)


test_designs_slow = pd.read_csv(
    TEST_DESIGNS_SLOW
)


test_designs_fast = pd.read_csv(
    TEST_DESIGNS_FAST
)


print(
    f"Typical: {test_designs_typical.shape}"
)


print(
    f"Slow:    {test_designs_slow.shape}"
)


print(
    f"Fast:    {test_designs_fast.shape}"
)


# ============================================================
# CREAR TEST DESIGNS
# ============================================================

test_designs = crear_dataset_multicorner(
    test_designs_typical,
    test_designs_slow,
    test_designs_fast,
)


print(
    f"Test Designs final: "
    f"{test_designs.shape}"
)


# ============================================================
# PREPARAR TEST LABELS
# ============================================================

X_test_labels = test_labels[
    columnas_features
].copy()


X_test_labels = X_test_labels.apply(
    pd.to_numeric,
    errors="coerce",
)


y_test_labels_slow = test_labels[
    "Delay_Slow"
]


y_test_labels_fast = test_labels[
    "Delay_Fast"
]


# ============================================================
# PREPARAR TEST DESIGNS
# ============================================================

X_test_designs = test_designs[
    columnas_features
].copy()


X_test_designs = X_test_designs.apply(
    pd.to_numeric,
    errors="coerce",
)


y_test_designs_slow = test_designs[
    "Delay_Slow"
]


y_test_designs_fast = test_designs[
    "Delay_Fast"
]


# ============================================================
# BASELINE EN TEST
# ============================================================

print("\n" + "=" * 70)
print("BASELINE EN TEST LABELS Y TEST DESIGNS")
print("=" * 70)


# ============================================================
# TEST LABELS
# ============================================================

baseline_test_labels_slow = (
    test_labels["Delay_Typical"].values
)


baseline_test_labels_fast = (
    test_labels["Delay_Typical"].values
)


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Test Labels",
        "Slow",
        y_test_labels_slow,
        baseline_test_labels_slow,
    )
)


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Test Labels",
        "Fast",
        y_test_labels_fast,
        baseline_test_labels_fast,
    )
)


# ============================================================
# TEST DESIGNS
# ============================================================

baseline_test_designs_slow = (
    test_designs["Delay_Typical"].values
)


baseline_test_designs_fast = (
    test_designs["Delay_Typical"].values
)


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Test Designs",
        "Slow",
        y_test_designs_slow,
        baseline_test_designs_slow,
    )
)


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Test Designs",
        "Fast",
        y_test_designs_fast,
        baseline_test_designs_fast,
    )
)


# ============================================================
# RIDGE FINAL
# ============================================================

print("\n" + "=" * 70)
print("RIDGE FINAL")
print("=" * 70)


modelo_ridge_final_slow = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "scaler",
            StandardScaler(),
        ),

        (
            "modelo",
            Ridge(
                alpha=1.0
            ),
        ),

    ]
)


modelo_ridge_final_fast = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "scaler",
            StandardScaler(),
        ),

        (
            "modelo",
            Ridge(
                alpha=1.0
            ),
        ),

    ]
)


modelo_ridge_final_slow.fit(
    X,
    y_slow,
)


modelo_ridge_final_fast.fit(
    X,
    y_fast,
)


# ============================================================
# RIDGE TEST LABELS
# ============================================================

pred_ridge_labels_slow = (
    modelo_ridge_final_slow.predict(
        X_test_labels
    )
)


pred_ridge_labels_fast = (
    modelo_ridge_final_fast.predict(
        X_test_labels
    )
)


resultados.append(
    crear_resultado(
        "Ridge",
        "alpha=1.0",
        "Test Labels",
        "Slow",
        y_test_labels_slow,
        pred_ridge_labels_slow,
    )
)


resultados.append(
    crear_resultado(
        "Ridge",
        "alpha=1.0",
        "Test Labels",
        "Fast",
        y_test_labels_fast,
        pred_ridge_labels_fast,
    )
)


# ============================================================
# RIDGE TEST DESIGNS
# ============================================================

pred_ridge_designs_slow = (
    modelo_ridge_final_slow.predict(
        X_test_designs
    )
)


pred_ridge_designs_fast = (
    modelo_ridge_final_fast.predict(
        X_test_designs
    )
)


resultados.append(
    crear_resultado(
        "Ridge",
        "alpha=1.0",
        "Test Designs",
        "Slow",
        y_test_designs_slow,
        pred_ridge_designs_slow,
    )
)


resultados.append(
    crear_resultado(
        "Ridge",
        "alpha=1.0",
        "Test Designs",
        "Fast",
        y_test_designs_fast,
        pred_ridge_designs_fast,
    )
)


# ============================================================
# RANDOM FOREST FINAL
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST FINAL")
print("=" * 70)


print(
    "\nConfiguracion Slow:"
)

print(
    mejor_config_slow
)


print(
    "\nConfiguracion Fast:"
)

print(
    mejor_config_fast
)


modelo_final_slow = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "modelo",
            RandomForestRegressor(

                n_estimators=mejor_config_slow[
                    "trees"
                ],

                max_depth=mejor_config_slow[
                    "depth"
                ],

                min_samples_leaf=mejor_config_slow[
                    "leaf"
                ],

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
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "modelo",
            RandomForestRegressor(

                n_estimators=mejor_config_fast[
                    "trees"
                ],

                max_depth=mejor_config_fast[
                    "depth"
                ],

                min_samples_leaf=mejor_config_fast[
                    "leaf"
                ],

                random_state=42,

                n_jobs=-1,

            ),
        ),

    ]
)


# ============================================================
# ENTRENAR CON TODO TRAIN
# ============================================================

modelo_final_slow.fit(
    X,
    y_slow,
)


modelo_final_fast.fit(
    X,
    y_fast,
)


# ============================================================
# EVALUAR RANDOM FOREST EN TRAIN
# ============================================================

pred_final_train_slow = (
    modelo_final_slow.predict(
        X
    )
)


pred_final_train_fast = (
    modelo_final_fast.predict(
        X
    )
)


resultados.append(
    crear_resultado(
        "Random Forest",
        mejor_slow["Experimento"],
        "Train",
        "Slow",
        y_slow,
        pred_final_train_slow,
    )
)


resultados.append(
    crear_resultado(
        "Random Forest",
        mejor_fast["Experimento"],
        "Train",
        "Fast",
        y_fast,
        pred_final_train_fast,
    )
)


# ============================================================
# RANDOM FOREST TEST LABELS
# ============================================================

pred_rf_labels_slow = (
    modelo_final_slow.predict(
        X_test_labels
    )
)


pred_rf_labels_fast = (
    modelo_final_fast.predict(
        X_test_labels
    )
)


resultados.append(
    crear_resultado(
        "Random Forest",
        mejor_slow["Experimento"],
        "Test Labels",
        "Slow",
        y_test_labels_slow,
        pred_rf_labels_slow,
    )
)


resultados.append(
    crear_resultado(
        "Random Forest",
        mejor_fast["Experimento"],
        "Test Labels",
        "Fast",
        y_test_labels_fast,
        pred_rf_labels_fast,
    )
)


# ============================================================
# RANDOM FOREST TEST DESIGNS
# ============================================================

pred_rf_designs_slow = (
    modelo_final_slow.predict(
        X_test_designs
    )
)


pred_rf_designs_fast = (
    modelo_final_fast.predict(
        X_test_designs
    )
)


resultados.append(
    crear_resultado(
        "Random Forest",
        mejor_slow["Experimento"],
        "Test Designs",
        "Slow",
        y_test_designs_slow,
        pred_rf_designs_slow,
    )
)


resultados.append(
    crear_resultado(
        "Random Forest",
        mejor_fast["Experimento"],
        "Test Designs",
        "Fast",
        y_test_designs_fast,
        pred_rf_designs_fast,
    )
)


# ============================================================
# TABLA FINAL
# ============================================================

resultados_completos = pd.DataFrame(
    resultados
)


print("\n" + "=" * 70)
print("TABLA FINAL DE RESULTADOS")
print("=" * 70)


print(
    resultados_completos.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# GUARDAR METRICAS
# ============================================================

ruta_metricas = (
    RESULTS_DIR
    / "metricas_objetivo3.csv"
)


resultados_completos.to_csv(
    ruta_metricas,
    index=False,
)


# ============================================================
# GUARDAR EXPERIMENTOS RANDOM FOREST
# ============================================================

ruta_experimentos = (
    RESULTS_DIR
    / "experimentos_random_forest.csv"
)


pd.DataFrame(
    resultados_experimentos_rf
).to_csv(
    ruta_experimentos,
    index=False,
)


# ============================================================
# GUARDAR PREDICCIONES TEST LABELS
# ============================================================

predicciones_test_labels = test_labels[
    [
        "row_id",
        "Delay_Typical",
        "Delay_Slow",
        "Delay_Fast",
    ]
].copy()


predicciones_test_labels[
    "Baseline_Slow"
] = baseline_test_labels_slow


predicciones_test_labels[
    "Baseline_Fast"
] = baseline_test_labels_fast


predicciones_test_labels[
    "Predicted_Ridge_Slow"
] = pred_ridge_labels_slow


predicciones_test_labels[
    "Predicted_Ridge_Fast"
] = pred_ridge_labels_fast


predicciones_test_labels[
    "Predicted_RF_Slow"
] = pred_rf_labels_slow


predicciones_test_labels[
    "Predicted_RF_Fast"
] = pred_rf_labels_fast


ruta_predicciones_labels = (
    RESULTS_DIR
    / "predicciones_test_labels.csv"
)


predicciones_test_labels.to_csv(
    ruta_predicciones_labels,
    index=False,
)


# ============================================================
# GUARDAR PREDICCIONES TEST DESIGNS
# ============================================================

predicciones_test_designs = test_designs[
    [
        "row_id",
        "Delay_Typical",
        "Delay_Slow",
        "Delay_Fast",
    ]
].copy()


predicciones_test_designs[
    "Baseline_Slow"
] = baseline_test_designs_slow


predicciones_test_designs[
    "Baseline_Fast"
] = baseline_test_designs_fast


predicciones_test_designs[
    "Predicted_Ridge_Slow"
] = pred_ridge_designs_slow


predicciones_test_designs[
    "Predicted_Ridge_Fast"
] = pred_ridge_designs_fast


predicciones_test_designs[
    "Predicted_RF_Slow"
] = pred_rf_designs_slow


predicciones_test_designs[
    "Predicted_RF_Fast"
] = pred_rf_designs_fast


ruta_predicciones_designs = (
    RESULTS_DIR
    / "predicciones_test_designs.csv"
)


predicciones_test_designs.to_csv(
    ruta_predicciones_designs,
    index=False,
)


# ============================================================
# GRAFICA:
# TYPICAL VS SLOW REAL
# ============================================================

print("\n" + "=" * 70)
print("GENERANDO GRAFICAS")
print("=" * 70)


def guardar_grafica_comparacion(
    x,
    y,
    titulo,
    xlabel,
    ylabel,
    nombre_archivo,
):

    plt.figure(
        figsize=(8, 7)
    )

    plt.scatter(
        x,
        y,
        s=8,
        alpha=0.35,
    )


    minimo = min(
        np.nanmin(x),
        np.nanmin(y),
    )


    maximo = max(
        np.nanmax(x),
        np.nanmax(y),
    )


    plt.plot(
        [minimo, maximo],
        [minimo, maximo],
        linestyle="--",
        linewidth=2,
    )


    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        titulo
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()


    ruta = (
        GRAFICAS_DIR
        / nombre_archivo
    )


    plt.savefig(
        ruta,
        dpi=200,
    )


    plt.close()


# ============================================================
# BASELINE: TYPICAL VS SLOW
# ============================================================

guardar_grafica_comparacion(

    test_designs["Delay_Typical"],

    test_designs["Delay_Slow"],

    "Delay Typical vs Delay Slow",

    "Delay Typical",

    "Delay Slow real",

    "01_typical_vs_slow_real.png",

)


# ============================================================
# BASELINE: TYPICAL VS FAST
# ============================================================

guardar_grafica_comparacion(

    test_designs["Delay_Typical"],

    test_designs["Delay_Fast"],

    "Delay Typical vs Delay Fast",

    "Delay Typical",

    "Delay Fast real",

    "02_typical_vs_fast_real.png",

)


# ============================================================
# PREDICCION VS REAL
# ============================================================

guardar_grafica_comparacion(

    y_test_designs_slow,

    pred_rf_designs_slow,

    "Random Forest: Slow predicho vs Slow real",

    "Slow real",

    "Slow predicho",

    "03_random_forest_slow_predicho_vs_real.png",

)


guardar_grafica_comparacion(

    y_test_designs_fast,

    pred_rf_designs_fast,

    "Random Forest: Fast predicho vs Fast real",

    "Fast real",

    "Fast predicho",

    "04_random_forest_fast_predicho_vs_real.png",

)


# ============================================================
# BASELINE VS RANDOM FOREST
# TEST DESIGNS - SLOW
# ============================================================

plt.figure(
    figsize=(8, 7)
)


plt.scatter(
    y_test_designs_slow,
    baseline_test_designs_slow,
    s=8,
    alpha=0.25,
    label="Baseline",
)


plt.scatter(
    y_test_designs_slow,
    pred_rf_designs_slow,
    s=8,
    alpha=0.25,
    label="Random Forest",
)


minimo = min(
    np.nanmin(y_test_designs_slow),
    np.nanmin(baseline_test_designs_slow),
    np.nanmin(pred_rf_designs_slow),
)


maximo = max(
    np.nanmax(y_test_designs_slow),
    np.nanmax(baseline_test_designs_slow),
    np.nanmax(pred_rf_designs_slow),
)


plt.plot(
    [minimo, maximo],
    [minimo, maximo],
    linestyle="--",
    linewidth=2,
)


plt.xlabel(
    "Slow real"
)

plt.ylabel(
    "Valor predicho"
)

plt.title(
    "Baseline vs Random Forest - Slow"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()


plt.savefig(
    GRAFICAS_DIR
    / "05_baseline_vs_random_forest_slow.png",
    dpi=200,
)


plt.close()


# ============================================================
# BASELINE VS RANDOM FOREST
# TEST DESIGNS - FAST
# ============================================================

plt.figure(
    figsize=(8, 7)
)


plt.scatter(
    y_test_designs_fast,
    baseline_test_designs_fast,
    s=8,
    alpha=0.25,
    label="Baseline",
)


plt.scatter(
    y_test_designs_fast,
    pred_rf_designs_fast,
    s=8,
    alpha=0.25,
    label="Random Forest",
)


minimo = min(
    np.nanmin(y_test_designs_fast),
    np.nanmin(baseline_test_designs_fast),
    np.nanmin(pred_rf_designs_fast),
)


maximo = max(
    np.nanmax(y_test_designs_fast),
    np.nanmax(baseline_test_designs_fast),
    np.nanmax(pred_rf_designs_fast),
)


plt.plot(
    [minimo, maximo],
    [minimo, maximo],
    linestyle="--",
    linewidth=2,
)


plt.xlabel(
    "Fast real"
)

plt.ylabel(
    "Valor predicho"
)

plt.title(
    "Baseline vs Random Forest - Fast"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()


plt.savefig(
    GRAFICAS_DIR
    / "06_baseline_vs_random_forest_fast.png",
    dpi=200,
)


plt.close()


# ============================================================
# GRAFICAS DE TEST LABELS
# ============================================================

guardar_grafica_comparacion(

    y_test_labels_slow,

    pred_rf_labels_slow,

    "Random Forest: Test Labels - Slow",

    "Slow real",

    "Slow predicho",

    "07_test_labels_slow_predicho_vs_real.png",

)


guardar_grafica_comparacion(

    y_test_labels_fast,

    pred_rf_labels_fast,

    "Random Forest: Test Labels - Fast",

    "Fast real",

    "Fast predicho",

    "08_test_labels_fast_predicho_vs_real.png",

)


# ============================================================
# GUARDAR MODELOS
# ============================================================

joblib.dump(
    modelo_final_slow,
    RESULTS_DIR
    / "modelo_random_forest_slow.joblib",
)


joblib.dump(
    modelo_final_fast,
    RESULTS_DIR
    / "modelo_random_forest_fast.joblib",
)


joblib.dump(
    modelo_ridge_final_slow,
    RESULTS_DIR
    / "modelo_ridge_slow.joblib",
)


joblib.dump(
    modelo_ridge_final_fast,
    RESULTS_DIR
    / "modelo_ridge_fast.joblib",
)


# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "=" * 70)
print("ARCHIVOS GENERADOS")
print("=" * 70)


print("\nMetricas:")

print(
    ruta_metricas
)


print("\nExperimentos Random Forest:")

print(
    ruta_experimentos
)


print("\nPredicciones Test Labels:")

print(
    ruta_predicciones_labels
)


print("\nPredicciones Test Designs:")

print(
    ruta_predicciones_designs
)


print("\nGraficas:")

print(
    GRAFICAS_DIR
)


print("\nModelos:")

print(
    RESULTS_DIR
)


print("\n" + "=" * 70)
print("PROCESO TERMINADO")
print("=" * 70)