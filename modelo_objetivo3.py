from pathlib import Path

import numpy as np
import pandas as pd

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

import joblib


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "resultados_objetivo3"

RESULTS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# ARCHIVOS DE ENTRENAMIENTO
# ============================================================

TRAIN_TYPICAL = (
    BASE_DIR / "treated_labels_train_typical.csv"
)

TRAIN_SLOW = (
    BASE_DIR / "treated_labels_train_slow.csv"
)

TRAIN_FAST = (
    BASE_DIR / "treated_labels_train_fast.csv"
)


# ============================================================
# ARCHIVOS TEST LABELS
# ============================================================

TEST_LABELS_TYPICAL = (
    BASE_DIR / "treated_labels_typical.csv"
)

TEST_LABELS_SLOW = (
    BASE_DIR / "treated_labels_slow.csv"
)

TEST_LABELS_FAST = (
    BASE_DIR / "treated_labels_fast.csv"
)


# ============================================================
# ARCHIVOS TEST DESIGNS
# ============================================================

TEST_DESIGNS_TYPICAL = (
    BASE_DIR / "treated_test_designs_typical.csv"
)

TEST_DESIGNS_SLOW = (
    BASE_DIR / "treated_test_designs_slow.csv"
)

TEST_DESIGNS_FAST = (
    BASE_DIR / "treated_test_designs_fast.csv"
)


# ============================================================
# FUNCIONES DE METRICAS
# ============================================================

def calcular_pearson(y_real, y_pred):

    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)

    if len(y_real) < 2:
        return np.nan

    if np.std(y_real) == 0:
        return np.nan

    if np.std(y_pred) == 0:
        return np.nan

    return np.corrcoef(
        y_real,
        y_pred
    )[0, 1]


def calcular_metricas(y_real, y_pred):

    mae = mean_absolute_error(
        y_real,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_real,
            y_pred
        )
    )

    r2 = r2_score(
        y_real,
        y_pred
    )

    pearson = calcular_pearson(
        y_real,
        y_pred
    )

    return (
        mae,
        rmse,
        r2,
        pearson,
    )


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
        y_pred
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
# LLAVE DE CORRESPONDENCIA
# ============================================================

columnas_llave = [
    "Previous_description",
    "Description",
]


# ============================================================
# VERIFICAR TRAIN
# ============================================================

print("\n" + "=" * 70)
print("VERIFICANDO CORRESPONDENCIA ENTRE ESQUINAS")
print("=" * 70)


for nombre, df in [
    ("Typical", train_typical),
    ("Slow", train_slow),
    ("Fast", train_fast),
]:

    if df.duplicated(
        columnas_llave
    ).any():

        raise ValueError(
            f"Hay segmentos duplicados en {nombre}."
        )


llaves_typical = set(
    zip(
        train_typical[
            "Previous_description"
        ],
        train_typical[
            "Description"
        ],
    )
)

llaves_slow = set(
    zip(
        train_slow[
            "Previous_description"
        ],
        train_slow[
            "Description"
        ],
    )
)

llaves_fast = set(
    zip(
        train_fast[
            "Previous_description"
        ],
        train_fast[
            "Description"
        ],
    )
)


if llaves_typical != llaves_slow:

    raise ValueError(
        "Typical y Slow no contienen "
        "los mismos segmentos."
    )


if llaves_typical != llaves_fast:

    raise ValueError(
        "Typical y Fast no contienen "
        "los mismos segmentos."
    )


print(
    "Los tres archivos contienen "
    "los mismos segmentos."
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


# ============================================================
# X Y
# ============================================================

X = datos_train[
    columnas_features
].copy()


X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


y_slow = datos_train[
    "Delay_Slow"
].copy()


y_fast = datos_train[
    "Delay_Fast"
].copy()


print(
    f"Cantidad de observaciones: "
    f"{len(X):,}"
)


# ============================================================
# DIVISION TRAIN / VALIDACION
# ============================================================

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
]

X_validacion = X.iloc[
    indices_validacion
]


y_slow_train = y_slow.iloc[
    indices_train
]

y_slow_validacion = y_slow.iloc[
    indices_validacion
]


y_fast_train = y_fast.iloc[
    indices_train
]

y_fast_validacion = y_fast.iloc[
    indices_validacion
]


print("\n" + "=" * 70)
print("DIVISION DE DATOS")
print("=" * 70)

print(
    f"Entrenamiento: {len(X_train):,}"
)

print(
    f"Validacion:    {len(X_validacion):,}"
)


# ============================================================
# RESULTADOS
# ============================================================

resultados = []


# ============================================================
# BASELINE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)

print(
    "Baseline: usar Delay_Typical "
    "como prediccion de Slow y Fast."
)


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

baseline_train = datos_train[
    "Delay_Typical"
].iloc[
    indices_train
].values


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Train",
        "Slow",
        y_slow_train,
        baseline_train,
    )
)


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Train",
        "Fast",
        y_fast_train,
        baseline_train,
    )
)


# ------------------------------------------------------------
# VALIDACION
# ------------------------------------------------------------

baseline_validacion = datos_train[
    "Delay_Typical"
].iloc[
    indices_validacion
].values


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Validacion",
        "Slow",
        y_slow_validacion,
        baseline_validacion,
    )
)


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Validacion",
        "Fast",
        y_fast_validacion,
        baseline_validacion,
    )
)


# ============================================================
# RIDGE
# ============================================================

print("\n" + "=" * 70)
print("RIDGE")
print("=" * 70)


def crear_ridge():

    return Pipeline(
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


ridge_slow = crear_ridge()

ridge_fast = crear_ridge()


ridge_slow.fit(
    X_train,
    y_slow_train
)


ridge_fast.fit(
    X_train,
    y_fast_train
)


# ------------------------------------------------------------
# PREDICCIONES TRAIN
# ------------------------------------------------------------

pred_ridge_slow_train = ridge_slow.predict(
    X_train
)

pred_ridge_fast_train = ridge_fast.predict(
    X_train
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
        "Train",
        "Fast",
        y_fast_train,
        pred_ridge_fast_train,
    )
)


# ------------------------------------------------------------
# PREDICCIONES VALIDACION
# ------------------------------------------------------------

pred_ridge_slow_validacion = ridge_slow.predict(
    X_validacion
)

pred_ridge_fast_validacion = ridge_fast.predict(
    X_validacion
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


configuraciones_rf = [
    {
        "n_estimators": 10,
        "max_depth": 20,
        "min_samples_leaf": 2,
    },

    {
        "n_estimators": 50,
        "max_depth": 20,
        "min_samples_leaf": 2,
    },

    {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 2,
    },

    {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 2,
    },
]


resultados_rf_validacion = []


for configuracion in configuraciones_rf:

    nombre_experimento = (
        f"trees={configuracion['n_estimators']}"
        f"_depth={configuracion['max_depth']}"
        f"_leaf={configuracion['min_samples_leaf']}"
    )


    print("\n" + "-" * 70)

    print(
        f"Random Forest: "
        f"{nombre_experimento}"
    )


    # --------------------------------------------------------
    # SLOW
    # --------------------------------------------------------

    modelo_slow = Pipeline(
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
                    n_estimators=configuracion[
                        "n_estimators"
                    ],

                    max_depth=configuracion[
                        "max_depth"
                    ],

                    min_samples_leaf=configuracion[
                        "min_samples_leaf"
                    ],

                    random_state=42,

                    n_jobs=-1,
                ),
            ),
        ]
    )


    modelo_slow.fit(
        X_train,
        y_slow_train
    )


    pred_slow_train = modelo_slow.predict(
        X_train
    )


    pred_slow_validacion = modelo_slow.predict(
        X_validacion
    )


    resultados.append(
        crear_resultado(
            "Random Forest",
            nombre_experimento,
            "Train",
            "Slow",
            y_slow_train,
            pred_slow_train,
        )
    )


    resultado_validacion_slow = crear_resultado(
        "Random Forest",
        nombre_experimento,
        "Validacion",
        "Slow",
        y_slow_validacion,
        pred_slow_validacion,
    )


    resultados.append(
        resultado_validacion_slow
    )


    resultados_rf_validacion.append(
        resultado_validacion_slow
    )


    # --------------------------------------------------------
    # FAST
    # --------------------------------------------------------

    modelo_fast = Pipeline(
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
                    n_estimators=configuracion[
                        "n_estimators"
                    ],

                    max_depth=configuracion[
                        "max_depth"
                    ],

                    min_samples_leaf=configuracion[
                        "min_samples_leaf"
                    ],

                    random_state=42,

                    n_jobs=-1,
                ),
            ),
        ]
    )


    modelo_fast.fit(
        X_train,
        y_fast_train
    )


    pred_fast_train = modelo_fast.predict(
        X_train
    )


    pred_fast_validacion = modelo_fast.predict(
        X_validacion
    )


    resultados.append(
        crear_resultado(
            "Random Forest",
            nombre_experimento,
            "Train",
            "Fast",
            y_fast_train,
            pred_fast_train,
        )
    )


    resultado_validacion_fast = crear_resultado(
        "Random Forest",
        nombre_experimento,
        "Validacion",
        "Fast",
        y_fast_validacion,
        pred_fast_validacion,
    )


    resultados.append(
        resultado_validacion_fast
    )


    resultados_rf_validacion.append(
        resultado_validacion_fast
    )


# ============================================================
# GRADIENT BOOSTING
# ============================================================

print("\n" + "=" * 70)
print("GRADIENT BOOSTING")
print("=" * 70)


def crear_gradient_boosting():

    return Pipeline(
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


gb_slow = crear_gradient_boosting()

gb_fast = crear_gradient_boosting()


gb_slow.fit(
    X_train,
    y_slow_train
)

gb_fast.fit(
    X_train,
    y_fast_train
)


pred_gb_slow_train = gb_slow.predict(
    X_train
)

pred_gb_fast_train = gb_fast.predict(
    X_train
)


pred_gb_slow_validacion = gb_slow.predict(
    X_validacion
)

pred_gb_fast_validacion = gb_fast.predict(
    X_validacion
)


resultados.append(
    crear_resultado(
        "Gradient Boosting",
        "trees=100_learning_rate=0.05_depth=3",
        "Train",
        "Slow",
        y_slow_train,
        pred_gb_slow_train,
    )
)


resultados.append(
    crear_resultado(
        "Gradient Boosting",
        "trees=100_learning_rate=0.05_depth=3",
        "Train",
        "Fast",
        y_fast_train,
        pred_gb_fast_train,
    )
)


resultados.append(
    crear_resultado(
        "Gradient Boosting",
        "trees=100_learning_rate=0.05_depth=3",
        "Validacion",
        "Slow",
        y_slow_validacion,
        pred_gb_slow_validacion,
    )
)


resultados.append(
    crear_resultado(
        "Gradient Boosting",
        "trees=100_learning_rate=0.05_depth=3",
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


print(
    resultados_df[
        resultados_df["Conjunto"]
        == "Validacion"
    ].sort_values(
        "RMSE"
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# SELECCIONAR MEJOR RANDOM FOREST
# ============================================================

rf_validacion_df = resultados_df[
    (resultados_df["Modelo"] == "Random Forest")
    &
    (resultados_df["Conjunto"] == "Validacion")
].copy()


mejor_rf_slow = rf_validacion_df[
    rf_validacion_df["Esquina"] == "Slow"
].sort_values(
    "RMSE"
).iloc[0]


mejor_rf_fast = rf_validacion_df[
    rf_validacion_df["Esquina"] == "Fast"
].sort_values(
    "RMSE"
).iloc[0]


print("\n" + "=" * 70)
print("MEJORES CONFIGURACIONES RANDOM FOREST")
print("=" * 70)


print(
    "\nSlow:"
)

print(
    mejor_rf_slow[
        [
            "Experimento",
            "RMSE",
            "R2",
            "Pearson",
        ]
    ].to_string()
)


print(
    "\nFast:"
)

print(
    mejor_rf_fast[
        [
            "Experimento",
            "RMSE",
            "R2",
            "Pearson",
        ]
    ].to_string()
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
# FUNCION PARA CONSTRUIR TEST MULTICORNER
# ============================================================

def construir_test_multicorner(
    typical,
    slow,
    fast,
):

    for nombre, df in [
        ("Typical", typical),
        ("Slow", slow),
        ("Fast", fast),
    ]:

        if df.duplicated(
            columnas_llave
        ).any():

            raise ValueError(
                f"Hay duplicados en test {nombre}."
            )


    slow_delay = slow[
        columnas_llave + ["Delay"]
    ].rename(
        columns={
            "Delay": "Delay_Slow"
        }
    )


    fast_delay = fast[
        columnas_llave + ["Delay"]
    ].rename(
        columns={
            "Delay": "Delay_Fast"
        }
    )


    resultado = typical.rename(
        columns={
            "Delay": "Delay_Typical"
        }
    ).copy()


    resultado = resultado.merge(
        slow_delay,
        on=columnas_llave,
        how="inner",
        validate="one_to_one",
    )


    resultado = resultado.merge(
        fast_delay,
        on=columnas_llave,
        how="inner",
        validate="one_to_one",
    )


    return resultado


# ============================================================
# CONSTRUIR TEST LABELS
# ============================================================

test_labels = construir_test_multicorner(
    test_labels_typical,
    test_labels_slow,
    test_labels_fast,
)


print(
    f"\nTest Labels final: "
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
# CONSTRUIR TEST DESIGNS
# ============================================================

test_designs = construir_test_multicorner(
    test_designs_typical,
    test_designs_slow,
    test_designs_fast,
)


print(
    f"\nTest Designs final: "
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
    errors="coerce"
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
    errors="coerce"
)


y_test_designs_slow = test_designs[
    "Delay_Slow"
]

y_test_designs_fast = test_designs[
    "Delay_Fast"
]


# ============================================================
# BASELINE TEST LABELS Y TEST DESIGNS
# ============================================================

print("\n" + "=" * 70)
print("BASELINE EN TEST LABELS Y TEST DESIGNS")
print("=" * 70)


# ------------------------------------------------------------
# TEST LABELS
# ------------------------------------------------------------

baseline_labels = test_labels[
    "Delay_Typical"
].values


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Test Labels",
        "Slow",
        y_test_labels_slow,
        baseline_labels,
    )
)


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Test Labels",
        "Fast",
        y_test_labels_fast,
        baseline_labels,
    )
)


# ------------------------------------------------------------
# TEST DESIGNS
# ------------------------------------------------------------

baseline_designs = test_designs[
    "Delay_Typical"
].values


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Test Designs",
        "Slow",
        y_test_designs_slow,
        baseline_designs,
    )
)


resultados.append(
    crear_resultado(
        "Baseline",
        "Typical como prediccion",
        "Test Designs",
        "Fast",
        y_test_designs_fast,
        baseline_designs,
    )
)


# ============================================================
# ENTRENAR RIDGE FINAL
# ============================================================

print("\n" + "=" * 70)
print("RIDGE FINAL")
print("=" * 70)


ridge_final_slow = crear_ridge()

ridge_final_fast = crear_ridge()


ridge_final_slow.fit(
    X,
    y_slow
)

ridge_final_fast.fit(
    X,
    y_fast
)


# ============================================================
# RIDGE TEST LABELS
# ============================================================

pred_ridge_labels_slow = ridge_final_slow.predict(
    X_test_labels
)

pred_ridge_labels_fast = ridge_final_fast.predict(
    X_test_labels
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

pred_ridge_designs_slow = ridge_final_slow.predict(
    X_test_designs
)

pred_ridge_designs_fast = ridge_final_fast.predict(
    X_test_designs
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
# OBTENER PARAMETROS DEL MEJOR RF
# ============================================================

def obtener_parametros_rf(texto):

    partes = texto.split("_")

    parametros = {}

    for parte in partes:

        if "=" in parte:

            clave, valor = parte.split("=")

            parametros[clave] = int(valor)

    return parametros


param_slow = obtener_parametros_rf(
    mejor_rf_slow["Experimento"]
)


param_fast = obtener_parametros_rf(
    mejor_rf_fast["Experimento"]
)


# ============================================================
# ENTRENAR RF FINAL SLOW
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST FINAL")
print("=" * 70)


print(
    "\nConfiguracion Slow:"
)

print(
    param_slow
)


print(
    "\nConfiguracion Fast:"
)

print(
    param_fast
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
                n_estimators=param_slow[
                    "trees"
                ],

                max_depth=param_slow[
                    "depth"
                ],

                min_samples_leaf=param_slow[
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
                n_estimators=param_fast[
                    "trees"
                ],

                max_depth=param_fast[
                    "depth"
                ],

                min_samples_leaf=param_fast[
                    "leaf"
                ],

                random_state=42,

                n_jobs=-1,
            ),
        ),
    ]
)


# Entrenar con TODO el train

modelo_final_slow.fit(
    X,
    y_slow
)


modelo_final_fast.fit(
    X,
    y_fast
)


# ============================================================
# RF TEST LABELS
# ============================================================

pred_rf_labels_slow = modelo_final_slow.predict(
    X_test_labels
)

pred_rf_labels_fast = modelo_final_fast.predict(
    X_test_labels
)


resultados.append(
    crear_resultado(
        "Random Forest",
        mejor_rf_slow["Experimento"],
        "Test Labels",
        "Slow",
        y_test_labels_slow,
        pred_rf_labels_slow,
    )
)


resultados.append(
    crear_resultado(
        "Random Forest",
        mejor_rf_fast["Experimento"],
        "Test Labels",
        "Fast",
        y_test_labels_fast,
        pred_rf_labels_fast,
    )
)


# ============================================================
# RF TEST DESIGNS
# ============================================================

pred_rf_designs_slow = modelo_final_slow.predict(
    X_test_designs
)

pred_rf_designs_fast = modelo_final_fast.predict(
    X_test_designs
)


resultados.append(
    crear_resultado(
        "Random Forest",
        mejor_rf_slow["Experimento"],
        "Test Designs",
        "Slow",
        y_test_designs_slow,
        pred_rf_designs_slow,
    )
)


resultados.append(
    crear_resultado(
        "Random Forest",
        mejor_rf_fast["Experimento"],
        "Test Designs",
        "Fast",
        y_test_designs_fast,
        pred_rf_designs_fast,
    )
)


# ============================================================
# GUARDAR TODAS LAS METRICAS
# ============================================================

resultados_df = pd.DataFrame(
    resultados
)


ruta_metricas = (
    RESULTS_DIR
    / "metricas_objetivo3.csv"
)


resultados_df.to_csv(
    ruta_metricas,
    index=False
)


# ============================================================
# TABLA DE EXPERIMENTOS RF
# ============================================================

tabla_experimentos = resultados_df[
    (
        resultados_df["Modelo"]
        == "Random Forest"
    )
    &
    (
        resultados_df["Conjunto"]
        == "Validacion"
    )
].copy()


ruta_experimentos = (
    RESULTS_DIR
    / "experimentos_random_forest.csv"
)


tabla_experimentos.to_csv(
    ruta_experimentos,
    index=False
)


# ============================================================
# GUARDAR PREDICCIONES TEST LABELS
# ============================================================

predicciones_labels = test_labels[
    [
        "Delay_Typical",
        "Delay_Slow",
        "Delay_Fast",
    ]
].copy()


predicciones_labels[
    "Baseline_Slow"
] = baseline_labels


predicciones_labels[
    "Baseline_Fast"
] = baseline_labels


predicciones_labels[
    "Ridge_Slow"
] = pred_ridge_labels_slow


predicciones_labels[
    "Ridge_Fast"
] = pred_ridge_labels_fast


predicciones_labels[
    "RF_Slow"
] = pred_rf_labels_slow


predicciones_labels[
    "RF_Fast"
] = pred_rf_labels_fast


ruta_predicciones_labels = (
    RESULTS_DIR
    / "predicciones_test_labels.csv"
)


predicciones_labels.to_csv(
    ruta_predicciones_labels,
    index=False
)


# ============================================================
# GUARDAR PREDICCIONES TEST DESIGNS
# ============================================================

predicciones_designs = test_designs[
    [
        "row_id",
        "Delay_Typical",
        "Delay_Slow",
        "Delay_Fast",
    ]
].copy()


predicciones_designs[
    "Baseline_Slow"
] = baseline_designs


predicciones_designs[
    "Baseline_Fast"
] = baseline_designs


predicciones_designs[
    "Ridge_Slow"
] = pred_ridge_designs_slow


predicciones_designs[
    "Ridge_Fast"
] = pred_ridge_designs_fast


predicciones_designs[
    "RF_Slow"
] = pred_rf_designs_slow


predicciones_designs[
    "RF_Fast"
] = pred_rf_designs_fast


ruta_predicciones_designs = (
    RESULTS_DIR
    / "predicciones_test_designs.csv"
)


predicciones_designs.to_csv(
    ruta_predicciones_designs,
    index=False
)


# ============================================================
# GUARDAR MODELOS
# ============================================================

joblib.dump(
    ridge_final_slow,
    RESULTS_DIR
    / "modelo_ridge_slow.joblib"
)


joblib.dump(
    ridge_final_fast,
    RESULTS_DIR
    / "modelo_ridge_fast.joblib"
)


joblib.dump(
    modelo_final_slow,
    RESULTS_DIR
    / "modelo_random_forest_slow.joblib"
)


joblib.dump(
    modelo_final_fast,
    RESULTS_DIR
    / "modelo_random_forest_fast.joblib"
)


# ============================================================
# MOSTRAR TABLA FINAL
# ============================================================

print("\n" + "=" * 70)
print("TABLA FINAL DE RESULTADOS")
print("=" * 70)


print(
    resultados_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# RUTAS DE SALIDA
# ============================================================

print("\n" + "=" * 70)
print("ARCHIVOS GENERADOS")
print("=" * 70)


print(
    f"\nMetricas:"
)

print(
    ruta_metricas
)


print(
    f"\nExperimentos Random Forest:"
)

print(
    ruta_experimentos
)


print(
    f"\nPredicciones Test Labels:"
)

print(
    ruta_predicciones_labels
)


print(
    f"\nPredicciones Test Designs:"
)

print(
    ruta_predicciones_designs
)


print(
    "\nModelos:"
)

print(
    RESULTS_DIR
)


print("\n" + "=" * 70)
print("PROCESO TERMINADO")
print("=" * 70)