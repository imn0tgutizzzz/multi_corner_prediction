from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit(
        "No se encontro XGBoost.\n"
        "Instale con:\n"
        "python -m pip install xgboost"
    ) from exc


# ============================================================
# OBJETIVO 3 - MACHINE LEARNING MULTICORNER
#
# Se comparan dos estrategias:
#
# 1) DELAY DIRECTO
#       modelo -> Delay Slow/Fast
#
# 2) DELTA
#       modelo -> Delta Slow/Fast
#
#       Delta_Slow = Delay_Slow - Delay_Typical
#       Delta_Fast = Delay_Fast - Delay_Typical
#
#       Delay_pred = Delay_Typical + Delta_pred
#
# Ademas se comparan:
#   - Baseline
#   - Ridge
#   - Random Forest
#   - Gradient Boosting
#   - Hist Gradient Boosting
#   - XGBoost
#   - SVR
#
# Se evaluan:
#   - Train
#   - Validacion
#   - Test Labels
#   - Test Designs
#
# Metricas:
#   - MAE
#   - RMSE
#   - R2
#   - Pearson
#
# Tambien:
#   - cantidad de arboles
#   - feature importance
#   - graficas
#   - predicciones
#
# IMPORTANTE:
# Delay_Typical se utiliza como feature porque se supone conocido.
#
# Delta original NO se utiliza como feature para evitar leakage.
# ============================================================


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "resultados_objetivo3"
GRAFICAS_DIR = RESULTS_DIR / "graficas"
IMPORTANCIA_DIR = RESULTS_DIR / "importancia"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)
IMPORTANCIA_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42

# SVR puede ser muy lento con 60 mil muestras.
# Se utiliza una muestra para entrenarlo.
SVR_MAX_TRAIN = 12000

# Para evitar graficas gigantes.
PLOT_MAX_POINTS = 15000


# ============================================================
# COLUMNAS
# ============================================================

COLUMNAS_LLAVE = [
    "Previous_description",
    "Description",
]


# Estas columnas NO pueden entrar como features.
COLUMNAS_EXCLUIR = {
    "row_id",
    "Design",
    "Path #",
    "Previous_description",
    "Description",
    "Delta",
    "Delay_Slow",
    "Delay_Fast",
    "Delta_Slow",
    "Delta_Fast",
}


# ============================================================
# CARGAR Y ALINEAR LOS TRES CORNERS
# ============================================================

def cargar_multicorner(prefijo):

    archivos = {}

    print("\n" + "-" * 70)
    print(f"CARGANDO: {prefijo}")
    print("-" * 70)

    for esquina in ("typical", "slow", "fast"):

        ruta = BASE_DIR / f"{prefijo}_{esquina}.csv"

        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontro el archivo:\n{ruta}"
            )

        df = pd.read_csv(ruta)

        df.columns = df.columns.str.strip()

        archivos[esquina] = df

        print(
            f"{ruta.name}: "
            f"{df.shape[0]:,} filas x {df.shape[1]} columnas"
        )

    # --------------------------------------------------------
    # Determinar llaves
    # --------------------------------------------------------

    llaves = COLUMNAS_LLAVE.copy()

    # Si los tres archivos tienen Design,
    # tambien se usa Design como parte de la llave.
    if all("Design" in df.columns for df in archivos.values()):
        llaves = ["Design"] + llaves

    print("\nLlaves utilizadas:")
    for llave in llaves:
        print(" -", llave)

    # --------------------------------------------------------
    # Validaciones
    # --------------------------------------------------------

    for nombre, df in archivos.items():

        faltantes = set(llaves + ["Delay"]) - set(df.columns)

        if faltantes:
            raise ValueError(
                f"{nombre}: faltan columnas {faltantes}"
            )

        if df[llaves].isna().any().any():
            raise ValueError(
                f"{nombre}: hay llaves nulas"
            )

        if df.duplicated(llaves).any():
            duplicados = df.duplicated(llaves).sum()

            raise ValueError(
                f"{nombre}: hay {duplicados} llaves duplicadas"
            )

    # --------------------------------------------------------
    # Typical como base
    # --------------------------------------------------------

    datos = archivos["typical"].rename(
        columns={"Delay": "Delay_Typical"}
    ).copy()

    # --------------------------------------------------------
    # Agregar Slow y Fast
    # --------------------------------------------------------

    for nombre in ("slow", "fast"):

        etiqueta = archivos[nombre][
            llaves + ["Delay"]
        ].rename(
            columns={
                "Delay": f"Delay_{nombre.title()}"
            }
        )

        datos = datos.merge(
            etiqueta,
            on=llaves,
            how="left",
            validate="one_to_one",
            sort=False,
            indicator=True,
        )

        faltantes = (datos["_merge"] != "both").sum()

        datos = datos.drop(columns="_merge")

        if faltantes:
            raise ValueError(
                f"{prefijo}: "
                f"{faltantes} segmentos no tienen "
                f"correspondencia en {nombre}"
            )

    # --------------------------------------------------------
    # Convertir delays a numerico
    # --------------------------------------------------------

    for esquina in ("Typical", "Slow", "Fast"):

        columna = f"Delay_{esquina}"

        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce",
        )

        if not np.isfinite(datos[columna]).all():

            raise ValueError(
                f"{prefijo}: {columna} tiene "
                f"NaN o infinito"
            )

    # --------------------------------------------------------
    # Calcular Delta
    # --------------------------------------------------------

    datos["Delta_Slow"] = (
        datos["Delay_Slow"]
        - datos["Delay_Typical"]
    )

    datos["Delta_Fast"] = (
        datos["Delay_Fast"]
        - datos["Delay_Typical"]
    )

    print(
        f"\nSegmentos alineados: "
        f"{len(datos):,}"
    )

    return datos


# ============================================================
# METRICAS
# ============================================================

def metricas(y_real, y_pred):

    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if not np.isfinite(y_real).all():
        raise ValueError(
            "Hay valores no finitos en y_real"
        )

    if not np.isfinite(y_pred).all():
        raise ValueError(
            "Hay valores no finitos en y_pred"
        )

    pearson = np.nan

    if (
        len(y_real) > 1
        and np.std(y_real) > 0
        and np.std(y_pred) > 0
    ):
        pearson = np.corrcoef(
            y_real,
            y_pred
        )[0, 1]

    return {
        "MAE": mean_absolute_error(
            y_real,
            y_pred
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                y_real,
                y_pred
            )
        ),

        "R2": r2_score(
            y_real,
            y_pred
        ),

        "Pearson": pearson,
    }


# ============================================================
# REGISTRAR RESULTADOS
# ============================================================

def registrar(
    resultados,
    modelo,
    experimento,
    objetivo,
    conjunto,
    esquina,
    y_real,
    y_pred,
    n_entrenamiento=None,
    metrica_evaluada="Delay",
):

    resultados.append({

        "Modelo": modelo,

        "Experimento": experimento,

        "Objetivo": objetivo,

        "Conjunto": conjunto,

        "Esquina": esquina,

        "Metrica_evaluada": metrica_evaluada,

        "N_entrenamiento": n_entrenamiento,

        "N_evaluacion": len(y_real),

        **metricas(
            y_real,
            y_pred
        ),
    })


# ============================================================
# CREAR MODELO
# ============================================================

def crear_modelo(nombre, parametros):

    if nombre == "Ridge":

        pasos = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "modelo",
                Ridge(**parametros)
            ),
        ]

    elif nombre == "SVR":

        pasos = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "modelo",
                SVR(**parametros)
            ),
        ]

    elif nombre == "Random Forest":

        pasos = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "modelo",
                RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    **parametros
                )
            ),
        ]

    elif nombre == "Gradient Boosting":

        pasos = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "modelo",
                GradientBoostingRegressor(
                    random_state=RANDOM_STATE,
                    **parametros
                )
            ),
        ]

    elif nombre == "Hist Gradient Boosting":

        pasos = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "modelo",
                HistGradientBoostingRegressor(
                    random_state=RANDOM_STATE,
                    **parametros
                )
            ),
        ]

    elif nombre == "XGBoost":

        pasos = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "modelo",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                    **parametros
                )
            ),
        ]

    else:

        raise ValueError(
            f"Modelo desconocido: {nombre}"
        )

    return Pipeline(pasos)


# ============================================================
# EXPERIMENTOS
# ============================================================

def crear_experimentos():

    experimentos = []

    # --------------------------------------------------------
    # RIDGE
    # --------------------------------------------------------

    experimentos.append(
        (
            "Ridge",
            {
                "alpha": 1.0
            }
        )
    )

    # --------------------------------------------------------
    # GRADIENT BOOSTING
    # --------------------------------------------------------

    experimentos.append(
        (
            "Gradient Boosting",
            {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 3,
            }
        )
    )

    # --------------------------------------------------------
    # HIST GRADIENT BOOSTING
    # --------------------------------------------------------

    experimentos.append(
        (
            "Hist Gradient Boosting",
            {
                "max_iter": 200,
                "learning_rate": 0.05,
                "max_leaf_nodes": 31,
            }
        )
    )

    # --------------------------------------------------------
    # SVR
    # --------------------------------------------------------

    experimentos.append(
        (
            "SVR",
            {
                "kernel": "rbf",
                "C": 1.0,
                "epsilon": 0.1,
            }
        )
    )

    # --------------------------------------------------------
    # RANDOM FOREST - CANTIDAD DE ARBOLES
    # --------------------------------------------------------

    for n in (
        10,
        50,
        100,
        200,
        300,
        500,
    ):

        experimentos.append(
            (
                "Random Forest",
                {
                    "n_estimators": n,
                    "max_depth": 20,
                    "min_samples_leaf": 2,
                }
            )
        )

    # --------------------------------------------------------
    # RANDOM FOREST - PROFUNDIDAD
    # --------------------------------------------------------

    for profundidad in (
        10,
        30,
        None,
    ):

        experimentos.append(
            (
                "Random Forest",
                {
                    "n_estimators": 100,
                    "max_depth": profundidad,
                    "min_samples_leaf": 2,
                }
            )
        )

    # --------------------------------------------------------
    # RANDOM FOREST - MIN SAMPLES LEAF
    # --------------------------------------------------------

    for hoja in (
        1,
        4,
    ):

        experimentos.append(
            (
                "Random Forest",
                {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "min_samples_leaf": hoja,
                }
            )
        )

    # --------------------------------------------------------
    # XGBOOST - CANTIDAD DE ARBOLES
    # --------------------------------------------------------

    for n in (
        100,
        200,
        300,
        500,
    ):

        experimentos.append(
            (
                "XGBoost",
                {
                    "n_estimators": n,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                }
            )
        )

    return experimentos


# ============================================================
# PREPARAR FEATURES
# ============================================================

def preparar_X(
    datos,
    columnas_features
):

    faltantes = (
        set(columnas_features)
        - set(datos.columns)
    )

    if faltantes:

        raise ValueError(
            f"Faltan variables: {faltantes}"
        )

    X = datos[
        columnas_features
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if X.isna().all().any():

        columnas_vacias = (
            X.columns[
                X.isna().all()
            ].tolist()
        )

        raise ValueError(
            "Variables completamente vacias: "
            f"{columnas_vacias}"
        )

    return X


# ============================================================
# PREDICCION DEL TARGET
# ============================================================

def predecir_target(
    modelo,
    X
):

    return modelo.predict(X)


# ============================================================
# PREDICCION DEL DELAY FINAL
# ============================================================

def predecir_delay(
    modelo,
    X,
    objetivo
):

    pred = modelo.predict(X)

    if objetivo == "Delta":

        pred = (
            X["Delay_Typical"].to_numpy()
            + pred
        )

    return pred


# ============================================================
# GRAFICA REAL VS PREDICHO
# ============================================================

def guardar_grafica(
    y_real,
    y_pred,
    titulo,
    ruta,
    xlabel="Delay real",
    ylabel="Delay predicho",
):

    y_real = np.asarray(
        y_real,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    cantidad = min(
        len(y_real),
        PLOT_MAX_POINTS
    )

    idx = rng.choice(
        len(y_real),
        size=cantidad,
        replace=False,
    )

    lo = min(
        y_real.min(),
        y_pred.min()
    )

    hi = max(
        y_real.max(),
        y_pred.max()
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.scatter(
        y_real[idx],
        y_pred[idx],
        s=8,
        alpha=0.3,
    )

    # linea ideal y = x
    ax.plot(
        [lo, hi],
        [lo, hi],
        "--",
        color="black",
        linewidth=1.5,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        ruta,
        dpi=180
    )

    plt.close(fig)


# ============================================================
# GRAFICA BASELINE
# Typical vs real Slow/Fast
# ============================================================

def guardar_grafica_baseline(
    y_typical,
    y_real,
    esquina,
    conjunto,
    ruta,
):

    y_typical = np.asarray(
        y_typical,
        dtype=float
    )

    y_real = np.asarray(
        y_real,
        dtype=float
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    cantidad = min(
        len(y_real),
        PLOT_MAX_POINTS
    )

    idx = rng.choice(
        len(y_real),
        size=cantidad,
        replace=False,
    )

    lo = min(
        y_typical.min(),
        y_real.min()
    )

    hi = max(
        y_typical.max(),
        y_real.max()
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.scatter(
        y_typical[idx],
        y_real[idx],
        s=8,
        alpha=0.3,
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        "--",
        color="black",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Delay Typical"
    )

    ax.set_ylabel(
        f"Delay {esquina} real"
    )

    ax.set_title(
        f"{conjunto}: Typical vs {esquina}"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        ruta,
        dpi=180
    )

    plt.close(fig)


# ============================================================
# GRAFICA IMPORTANCIA
# ============================================================

def guardar_importancias(
    datos,
    ruta,
    titulo,
):

    datos = datos.sort_values(
        "Importancia",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        datos["Feature"],
        datos["Importancia"]
    )

    ax.set_xlabel(
        "Importancia relativa"
    )

    ax.set_title(
        titulo
    )

    fig.tight_layout()

    fig.savefig(
        ruta,
        dpi=180
    )

    plt.close(fig)


# ============================================================
# DIAGNOSTICO DE DELTAS
# ============================================================

def diagnostico_deltas(
    train,
    test_labels,
    test_designs,
):

    filas = []

    conjuntos = [
        ("Train", train),
        ("Test Labels", test_labels),
        ("Test Designs", test_designs),
    ]

    for nombre, datos in conjuntos:

        for esquina in (
            "Slow",
            "Fast",
        ):

            delta = datos[
                f"Delta_{esquina}"
            ].to_numpy(float)

            filas.append({

                "Conjunto": nombre,

                "Esquina": esquina,

                "N": len(delta),

                "Media_Delta": np.mean(delta),

                "Desv_Delta": np.std(delta),

                "Min_Delta": np.min(delta),

                "Mediana_Delta": np.median(delta),

                "Max_Delta": np.max(delta),

                # Baseline Delta = 0
                "RMSE_Delta_baseline": np.sqrt(
                    np.mean(delta ** 2)
                ),
            })

    df = pd.DataFrame(filas)

    df.to_csv(
        RESULTS_DIR
        / "diagnostico_distribucion_delta.csv",
        index=False,
    )

    return df


# ============================================================
# IMPORTANCIA DE VARIABLES
# ============================================================

def obtener_importancias(
    modelo,
    columnas_features,
):

    if "modelo" not in modelo.named_steps:
        return None

    estimador = (
        modelo.named_steps["modelo"]
    )

    if not hasattr(
        estimador,
        "feature_importances_"
    ):
        return None

    valores = (
        estimador.feature_importances_
    )

    if len(valores) != len(
        columnas_features
    ):
        raise ValueError(
            "La cantidad de importancias "
            "no coincide con las features."
        )

    df = pd.DataFrame({

        "Feature": columnas_features,

        "Importancia": valores,
    })

    df = df.sort_values(
        "Importancia",
        ascending=False
    ).reset_index(drop=True)

    return df


# ============================================================
# CREAR IMPORTANCIA RF
# ============================================================

def registrar_importancias(
    importancias,
    df_importancia,
    esquina,
    objetivo,
    modelo_nombre,
    experimento,
):

    if df_importancia is None:
        return

    for _, fila in (
        df_importancia.iterrows()
    ):

        importancias.append({

            "Esquina": esquina,

            "Objetivo": objetivo,

            "Modelo": modelo_nombre,

            "Experimento": experimento,

            "Feature": fila["Feature"],

            "Importancia": fila["Importancia"],
        })


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        "OBJETIVO 3 - "
        "MACHINE LEARNING MULTICORNER"
    )
    print("=" * 70)

    # ========================================================
    # CARGAR DATOS
    # ========================================================

    print("\nCARGANDO TRAIN")

    train = cargar_multicorner(
        "treated_labels_train"
    )

    print("\nCARGANDO TEST LABELS")

    test_labels = cargar_multicorner(
        "treated_labels"
    )

    print("\nCARGANDO TEST DESIGNS")

    test_designs = cargar_multicorner(
        "treated_test_designs"
    )

    conjuntos_test = {

        "Test Labels": test_labels,

        "Test Designs": test_designs,
    }

    # ========================================================
    # DIAGNOSTICO DELTA
    # ========================================================

    print("\n" + "=" * 70)
    print("DIAGNOSTICO DE DELTAS")
    print("=" * 70)

    diagnostico = diagnostico_deltas(
        train,
        test_labels,
        test_designs,
    )

    print(
        diagnostico.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}"
        )
    )

    # ========================================================
    # FEATURES
    # ========================================================

    columnas_features = [
        c
        for c in train.columns
        if c not in COLUMNAS_EXCLUIR
    ]

    if (
        "Delay_Typical"
        not in columnas_features
    ):

        raise ValueError(
            "Delay_Typical debe ser "
            "una variable de entrada."
        )

    print("\n" + "=" * 70)
    print("VARIABLES UTILIZADAS")
    print("=" * 70)

    for numero, columna in enumerate(
        columnas_features,
        start=1
    ):

        print(
            f"{numero:2d}. {columna}"
        )

    print(
        "\nTotal de features:",
        len(columnas_features)
    )

    pd.DataFrame({

        "Feature": columnas_features

    }).to_csv(
        RESULTS_DIR
        / "features_utilizadas.csv",
        index=False,
    )

    # ========================================================
    # X
    # ========================================================

    X = preparar_X(
        train,
        columnas_features
    )

    X_tests = {

        nombre: preparar_X(
            datos,
            columnas_features
        )

        for nombre, datos
        in conjuntos_test.items()
    }

    # ========================================================
    # TRAIN / VALIDACION
    # ========================================================

    indices_train, indices_validacion = (
        train_test_split(
            np.arange(len(train)),
            test_size=0.20,
            random_state=RANDOM_STATE,
        )
    )

    X_train = X.iloc[
        indices_train
    ]

    X_validacion = X.iloc[
        indices_validacion
    ]

    print("\n" + "=" * 70)
    print("DIVISION DE DATOS")
    print("=" * 70)

    print(
        f"Train:       {len(X_train):,}"
    )

    print(
        f"Validacion:  {len(X_validacion):,}"
    )

    # ========================================================
    # RESULTADOS
    # ========================================================

    resultados = []

    mejores_modelos = []

    importancias = []

    predicciones = {}

    # Guardaremos especificamente
    # el mejor RF de cada caso.
    mejores_rf = {}

    # ========================================================
    # BASELINE
    # ========================================================

    print("\n" + "=" * 70)
    print("BASELINE")
    print("=" * 70)

    print(
        "\nEl baseline utiliza directamente "
        "Delay_Typical como prediccion."
    )

    print(
        "No aprende con los datos."
    )

    print(
        "Sirve como referencia para saber "
        "si ML realmente mejora."
    )

    conjuntos_base = [

        (
            "Train",
            train.iloc[
                indices_train
            ],
        ),

        (
            "Validacion",
            train.iloc[
                indices_validacion
            ],
        ),

        (
            "Test Labels",
            test_labels,
        ),

        (
            "Test Designs",
            test_designs,
        ),
    ]

    for conjunto, datos in conjuntos_base:

        for esquina in (
            "Slow",
            "Fast",
        ):

            # ----------------------------------------------
            # Baseline de Delay
            # ----------------------------------------------

            registrar(
                resultados,

                "Baseline",

                "Typical como prediccion",

                "Delay",

                conjunto,

                esquina,

                datos[
                    f"Delay_{esquina}"
                ],

                datos[
                    "Delay_Typical"
                ],

                n_entrenamiento=None,

                metrica_evaluada="Delay",
            )

            # ----------------------------------------------
            # Baseline de Delta
            # Delta = 0
            # ----------------------------------------------

            registrar(
                resultados,

                "Baseline",

                "Delta=0",

                "Delta",

                conjunto,

                esquina,

                datos[
                    f"Delta_{esquina}"
                ],

                np.zeros(
                    len(datos)
                ),

                n_entrenamiento=None,

                metrica_evaluada="Delta",
            )

            # ----------------------------------------------
            # Grafica Typical vs real
            # ----------------------------------------------

            guardar_grafica_baseline(

                datos[
                    "Delay_Typical"
                ],

                datos[
                    f"Delay_{esquina}"
                ],

                esquina,

                conjunto,

                GRAFICAS_DIR
                / (
                    f"{conjunto.lower().replace(' ', '_')}"
                    f"_typical_vs_{esquina.lower()}.png"
                ),
            )

    # ========================================================
    # EXPERIMENTOS
    # ========================================================

    experimentos = crear_experimentos()

    print("\n" + "=" * 70)
    print(
        f"TOTAL DE CONFIGURACIONES: "
        f"{len(experimentos)}"
    )
    print("=" * 70)

    # ========================================================
    # LOOP PRINCIPAL
    # ========================================================

    for esquina in (
        "Slow",
        "Fast",
    ):

        for objetivo in (
            "Delay",
            "Delta",
        ):

            print("\n\n" + "=" * 70)

            print(
                f"ESQUINA: {esquina}"
            )

            print(
                f"OBJETIVO: {objetivo}"
            )

            print("=" * 70)

            columna_objetivo = (
                f"{objetivo}_{esquina}"
            )

            y_train = train[
                columna_objetivo
            ].iloc[
                indices_train
            ]

            y_validacion = train[
                columna_objetivo
            ].iloc[
                indices_validacion
            ]

            real_delay_train = train[
                f"Delay_{esquina}"
            ].iloc[
                indices_train
            ]

            real_delay_validacion = train[
                f"Delay_{esquina}"
            ].iloc[
                indices_validacion
            ]

            candidatos = []

            # =================================================
            # TODOS LOS EXPERIMENTOS
            # =================================================

            for numero_experimento, (
                nombre_modelo,
                parametros
            ) in enumerate(
                experimentos,
                start=1
            ):

                experimento = json.dumps(
                    parametros,
                    sort_keys=True
                )

                print(
                    "\n"
                    + "-" * 70
                )

                print(
                    f"Experimento "
                    f"{numero_experimento}/"
                    f"{len(experimentos)}"
                )

                print(
                    f"Modelo: {nombre_modelo}"
                )

                print(
                    f"Esquina: {esquina}"
                )

                print(
                    f"Objetivo: {objetivo}"
                )

                print(
                    f"Parametros: "
                    f"{experimento}"
                )

                modelo = crear_modelo(
                    nombre_modelo,
                    parametros
                )

                # --------------------------------------------
                # SVR usa muestra
                # --------------------------------------------

                if (
                    nombre_modelo == "SVR"
                    and len(X_train)
                    > SVR_MAX_TRAIN
                ):

                    rng = (
                        np.random.default_rng(
                            RANDOM_STATE
                        )
                    )

                    idx_sub = np.sort(
                        rng.choice(
                            len(X_train),
                            size=SVR_MAX_TRAIN,
                            replace=False,
                        )
                    )

                    X_ajuste = (
                        X_train.iloc[
                            idx_sub
                        ]
                    )

                    y_ajuste = (
                        y_train.iloc[
                            idx_sub
                        ]
                    )

                else:

                    X_ajuste = X_train
                    y_ajuste = y_train

                # --------------------------------------------
                # Entrenar
                # --------------------------------------------

                modelo.fit(
                    X_ajuste,
                    y_ajuste
                )

                # --------------------------------------------
                # Prediccion target
                # --------------------------------------------

                pred_target_train = (
                    predecir_target(
                        modelo,
                        X_train
                    )
                )

                pred_target_validacion = (
                    predecir_target(
                        modelo,
                        X_validacion
                    )
                )

                # --------------------------------------------
                # Prediccion Delay final
                # --------------------------------------------

                pred_delay_train = (
                    predecir_delay(
                        modelo,
                        X_train,
                        objetivo
                    )
                )

                pred_delay_validacion = (
                    predecir_delay(
                        modelo,
                        X_validacion,
                        objetivo
                    )
                )

                # --------------------------------------------
                # Target puro
                # --------------------------------------------

                registrar(
                    resultados,

                    nombre_modelo,

                    experimento,

                    objetivo,

                    "Train",

                    esquina,

                    y_train,

                    pred_target_train,

                    len(X_ajuste),

                    objetivo,
                )

                registrar(
                    resultados,

                    nombre_modelo,

                    experimento,

                    objetivo,

                    "Validacion",

                    esquina,

                    y_validacion,

                    pred_target_validacion,

                    len(X_ajuste),

                    objetivo,
                )

                # --------------------------------------------
                # Delay final
                # --------------------------------------------

                registrar(
                    resultados,

                    nombre_modelo,

                    experimento,

                    objetivo,

                    "Train",

                    esquina,

                    real_delay_train,

                    pred_delay_train,

                    len(X_ajuste),

                    "Delay reconstruido",
                )

                registrar(
                    resultados,

                    nombre_modelo,

                    experimento,

                    objetivo,

                    "Validacion",

                    esquina,

                    real_delay_validacion,

                    pred_delay_validacion,

                    len(X_ajuste),

                    "Delay reconstruido",
                )

                # --------------------------------------------
                # RMSE para seleccionar ganador
                # --------------------------------------------

                rmse_delay = metricas(
                    real_delay_validacion,
                    pred_delay_validacion,
                )["RMSE"]

                candidatos.append({

                    "RMSE_Delay": rmse_delay,

                    "Modelo": nombre_modelo,

                    "Parametros": parametros,

                    "Experimento": experimento,

                    "Modelo_entrenado": modelo,
                })

            # =================================================
            # ORDENAR CANDIDATOS
            # =================================================

            candidatos_ordenados = sorted(
                candidatos,
                key=lambda x:
                x["RMSE_Delay"]
            )

            # =================================================
            # GUARDAR TODOS LOS CANDIDATOS
            # =================================================

            candidatos_df = pd.DataFrame([

                {
                    "Esquina": esquina,

                    "Objetivo": objetivo,

                    "Modelo": c["Modelo"],

                    "Experimento": c["Experimento"],

                    "RMSE_Delay_Validacion":
                        c["RMSE_Delay"],
                }

                for c in candidatos_ordenados
            ])

            candidatos_df.to_csv(

                RESULTS_DIR
                / (
                    f"ranking_{esquina.lower()}_"
                    f"{objetivo.lower()}.csv"
                ),

                index=False,
            )

            # =================================================
            # MEJOR MODELO
            # =================================================

            mejor = candidatos_ordenados[0]

            mejor_rmse = (
                mejor["RMSE_Delay"]
            )

            mejor_nombre = (
                mejor["Modelo"]
            )

            mejores_parametros = (
                mejor["Parametros"]
            )

            mejor_experimento = (
                mejor["Experimento"]
            )

            print("\n" + "*" * 70)

            print(
                "MEJOR MODELO"
            )

            print(
                f"Esquina: {esquina}"
            )

            print(
                f"Objetivo: {objetivo}"
            )

            print(
                f"Modelo: {mejor_nombre}"
            )

            print(
                f"Parametros: "
                f"{mejor_experimento}"
            )

            print(
                f"RMSE Delay validacion: "
                f"{mejor_rmse:.6f}"
            )

            print("*" * 70)

            # =================================================
            # GUARDAR INFORMACION DEL GANADOR
            # =================================================

            mejores_modelos.append({

                "Esquina": esquina,

                "Objetivo": objetivo,

                "Modelo": mejor_nombre,

                "Experimento": (
                    mejor_experimento
                ),

                "RMSE_Delay_validacion":
                    mejor_rmse,
            })

            # =================================================
            # ENTRENAMIENTO FINAL
            # =================================================

            print(
                "\nEntrenando modelo final "
                "con todo Train..."
            )

            modelo_final = crear_modelo(
                mejor_nombre,
                mejores_parametros
            )

            y_total = train[
                columna_objetivo
            ]

            # Para SVR mantenemos la limitacion.
            if (
                mejor_nombre == "SVR"
                and len(X) > SVR_MAX_TRAIN
            ):

                rng = (
                    np.random.default_rng(
                        RANDOM_STATE
                    )
                )

                idx_sub = np.sort(
                    rng.choice(
                        len(X),
                        size=SVR_MAX_TRAIN,
                        replace=False,
                    )
                )

                X_final = X.iloc[
                    idx_sub
                ]

                y_final = y_total.iloc[
                    idx_sub
                ]

            else:

                X_final = X
                y_final = y_total

            modelo_final.fit(
                X_final,
                y_final
            )

            # =================================================
            # GUARDAR MODELO
            # =================================================

            nombre_archivo = (
                f"{esquina.lower()}_"
                f"{objetivo.lower()}"
            )

            joblib.dump(
                modelo_final,
                RESULTS_DIR
                / (
                    f"mejor_modelo_"
                    f"{nombre_archivo}.joblib"
                ),
            )

            # =================================================
            # TRAIN FINAL
            # =================================================

            pred_train_target = (
                predecir_target(
                    modelo_final,
                    X
                )
            )

            pred_train_delay = (
                predecir_delay(
                    modelo_final,
                    X,
                    objetivo
                )
            )

            registrar(
                resultados,

                mejor_nombre,

                mejor_experimento,

                f"{objetivo} (final)",

                "Train final",

                esquina,

                train[
                    columna_objetivo
                ],

                pred_train_target,

                len(X_final),

                objetivo,
            )

            registrar(
                resultados,

                mejor_nombre,

                mejor_experimento,

                f"{objetivo} (final)",

                "Train final",

                esquina,

                train[
                    f"Delay_{esquina}"
                ],

                pred_train_delay,

                len(X_final),

                "Delay reconstruido",
            )

            # =================================================
            # TEST LABELS / TEST DESIGNS
            # =================================================

            for conjunto, datos in (
                conjuntos_test.items()
            ):

                pred_target = (
                    predecir_target(
                        modelo_final,
                        X_tests[conjunto]
                    )
                )

                pred_delay = (
                    predecir_delay(
                        modelo_final,
                        X_tests[conjunto],
                        objetivo
                    )
                )

                # --------------------------------------------
                # Target
                # --------------------------------------------

                registrar(
                    resultados,

                    mejor_nombre,

                    mejor_experimento,

                    f"{objetivo} (final)",

                    conjunto,

                    esquina,

                    datos[
                        columna_objetivo
                    ],

                    pred_target,

                    len(X_final),

                    objetivo,
                )

                # --------------------------------------------
                # Delay final
                # --------------------------------------------

                registrar(
                    resultados,

                    mejor_nombre,

                    mejor_experimento,

                    f"{objetivo} (final)",

                    conjunto,

                    esquina,

                    datos[
                        f"Delay_{esquina}"
                    ],

                    pred_delay,

                    len(X_final),

                    "Delay reconstruido",
                )

                # --------------------------------------------
                # Guardar predicciones
                # --------------------------------------------

                predicciones[
                    (
                        conjunto,
                        esquina,
                        objetivo,
                    )
                ] = pred_delay

                # --------------------------------------------
                # Grafica real vs predicho
                # --------------------------------------------

                guardar_grafica(

                    datos[
                        f"Delay_{esquina}"
                    ],

                    pred_delay,

                    (
                        f"{conjunto}: "
                        f"{esquina} - "
                        f"{objetivo} - "
                        f"{mejor_nombre}"
                    ),

                    GRAFICAS_DIR
                    / (
                        f"{conjunto.lower().replace(' ', '_')}_"
                        f"{nombre_archivo}.png"
                    ),
                )

            # =================================================
            # FEATURE IMPORTANCE DEL GANADOR
            # =================================================

            importancia_ganador = (
                obtener_importancias(
                    modelo_final,
                    columnas_features
                )
            )

            registrar_importancias(

                importancias,

                importancia_ganador,

                esquina,

                objetivo,

                mejor_nombre,

                mejor_experimento,
            )

            # =================================================
            # FEATURE IMPORTANCE DEL MEJOR RF
            #
            # Esto es especialmente importante porque
            # el profesor pidio revisar feature_importances_
            # de Random Forest.
            # =================================================

            rf_candidatos = [
                c
                for c in candidatos_ordenados
                if c["Modelo"]
                == "Random Forest"
            ]

            if rf_candidatos:

                mejor_rf = rf_candidatos[0]

                mejores_rf[
                    (
                        esquina,
                        objetivo,
                    )
                ] = mejor_rf

                print(
                    "\nMejor Random Forest "
                    "para feature importance:"
                )

                print(
                    f"Modelo: "
                    f"{mejor_rf['Modelo']}"
                )

                print(
                    f"Parametros: "
                    f"{mejor_rf['Experimento']}"
                )

                print(
                    f"RMSE validacion: "
                    f"{mejor_rf['RMSE_Delay']:.6f}"
                )

                rf_modelo = crear_modelo(
                    "Random Forest",
                    mejor_rf["Parametros"]
                )

                rf_modelo.fit(
                    X_train,
                    y_train
                )

                rf_importancia = (
                    obtener_importancias(
                        rf_modelo,
                        columnas_features
                    )
                )

                registrar_importancias(

                    importancias,

                    rf_importancia,

                    esquina,

                    objetivo,

                    "Random Forest - mejor",
                    
                    mejor_rf[
                        "Experimento"
                    ],
                )

                # --------------------------------------------
                # Guardar CSV especifico
                # --------------------------------------------

                rf_importancia.to_csv(

                    IMPORTANCIA_DIR
                    / (
                        f"feature_importance_"
                        f"rf_mejor_"
                        f"{esquina.lower()}_"
                        f"{objetivo.lower()}.csv"
                    ),

                    index=False,
                )

                # --------------------------------------------
                # Grafica
                # --------------------------------------------

                guardar_importancias(

                    rf_importancia,

                    IMPORTANCIA_DIR
                    / (
                        f"feature_importance_"
                        f"rf_mejor_"
                        f"{esquina.lower()}_"
                        f"{objetivo.lower()}.png"
                    ),

                    (
                        f"Feature Importance - "
                        f"RF mejor - "
                        f"{esquina} - "
                        f"{objetivo}"
                    ),
                )

    # ========================================================
    # DATAFRAME GENERAL DE METRICAS
    # ========================================================

    metricas_df = pd.DataFrame(
        resultados
    )

    metricas_df.to_csv(

        RESULTS_DIR
        / "metricas_objetivo3.csv",

        index=False,
    )

    # ========================================================
    # MEJORES MODELOS
    # ========================================================

    mejores_df = pd.DataFrame(
        mejores_modelos
    )

    mejores_df.to_csv(

        RESULTS_DIR
        / "mejores_modelos.csv",

        index=False,
    )

    # ========================================================
    # EXPERIMENTOS DE VALIDACION
    # ========================================================

    validacion_df = metricas_df[

        (
            metricas_df["Conjunto"]
            == "Validacion"
        )

        &

        (
            metricas_df["Metrica_evaluada"]
            == "Delay reconstruido"
        )

    ].copy()

    validacion_df.to_csv(

        RESULTS_DIR
        / "experimentos_validacion.csv",

        index=False,
    )

    # ========================================================
    # IMPORTANCIAS
    # ========================================================

    importancias_df = pd.DataFrame(
        importancias
    )

    importancias_df.to_csv(

        RESULTS_DIR
        / "importancia_variables.csv",

        index=False,
    )

    # ========================================================
    # GRAFICAS DE IMPORTANCIA
    # ========================================================

    if not importancias_df.empty:

        for (
            esquina,
            objetivo,
            modelo,
        ), grupo in importancias_df.groupby(
            [
                "Esquina",
                "Objetivo",
                "Modelo",
            ]
        ):

            nombre_seguro = (
                modelo
                .lower()
                .replace(" ", "_")
                .replace("/", "_")
            )

            nombre = (
                f"importancia_"
                f"{esquina.lower()}_"
                f"{objetivo.lower()}_"
                f"{nombre_seguro}.png"
            )

            guardar_importancias(

                grupo,

                GRAFICAS_DIR
                / nombre,

                (
                    f"Importancia: "
                    f"{esquina}, "
                    f"{objetivo}, "
                    f"{modelo}"
                ),
            )

    # ========================================================
    # PREDICCIONES DE TEST
    # ========================================================

    for conjunto, datos in (
        conjuntos_test.items()
    ):

        columnas = [
            "Delay_Typical",
            "Delay_Slow",
            "Delay_Fast",
        ]

        for columna in (
            "row_id",
            "Design",
            "Previous_description",
            "Description",
        ):

            if columna in datos.columns:

                columnas.insert(
                    0,
                    columna
                )

        pred_df = datos[
            columnas
        ].copy()

        for esquina in (
            "Slow",
            "Fast",
        ):

            # --------------------------------------------
            # Baseline
            # --------------------------------------------

            pred_df[
                f"Baseline_{esquina}"
            ] = (
                datos[
                    "Delay_Typical"
                ].to_numpy()
            )

            # --------------------------------------------
            # Delay / Delta
            # --------------------------------------------

            for objetivo in (
                "Delay",
                "Delta",
            ):

                pred_df[
                    f"Predicted_{esquina}_{objetivo}"
                ] = predicciones[
                    (
                        conjunto,
                        esquina,
                        objetivo,
                    )
                ]

        pred_df.to_csv(

            RESULTS_DIR
            / (
                "predicciones_"
                f"{conjunto.lower().replace(' ', '_')}.csv"
            ),

            index=False,
        )

    # ========================================================
    # TABLA DE 4 CONJUNTOS
    #
    # Esta es especialmente util para la reunion.
    # ========================================================

    filas_tabla = []

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    for esquina in (
        "Slow",
        "Fast",
    ):

        for conjunto in (
            "Train",
            "Validacion",
            "Test Labels",
            "Test Designs",
        ):

            filtro = (

                (metricas_df["Modelo"] == "Baseline")

                &

                (
                    metricas_df["Objetivo"]
                    == "Delay"
                )

                &

                (
                    metricas_df["Conjunto"]
                    == conjunto
                )

                &

                (
                    metricas_df["Esquina"]
                    == esquina
                )

                &

                (
                    metricas_df[
                        "Metrica_evaluada"
                    ]
                    == "Delay"
                )
            )

            encontrados = (
                metricas_df.loc[
                    filtro
                ]
            )

            if len(encontrados):

                fila = encontrados.iloc[0]

                filas_tabla.append({

                    "Tipo": "Baseline",

                    "Esquina": esquina,

                    "Objetivo": "Delay",

                    "Modelo": "Baseline",

                    "Conjunto": conjunto,

                    "MAE": fila["MAE"],

                    "RMSE": fila["RMSE"],

                    "R2": fila["R2"],

                    "Pearson": fila["Pearson"],
                })

    # --------------------------------------------------------
    # Ganadores
    # --------------------------------------------------------

    for _, mejor in (
        mejores_df.iterrows()
    ):

        esquina = mejor[
            "Esquina"
        ]

        objetivo = mejor[
            "Objetivo"
        ]

        modelo = mejor[
            "Modelo"
        ]

        experimento = mejor[
            "Experimento"
        ]

        conjuntos_resumen = [
            "Train final",
            "Validacion",
            "Test Labels",
            "Test Designs",
        ]

        for conjunto in (
            conjuntos_resumen
        ):

            filtro = (

                (metricas_df["Modelo"] == modelo)

                &

                (
                    metricas_df[
                        "Experimento"
                    ] == experimento
                )

                &

                (
                    metricas_df[
                        "Objetivo"
                    ]
                    == f"{objetivo} (final)"
                )

                &

                (
                    metricas_df[
                        "Conjunto"
                    ]
                    == conjunto
                )

                &

                (
                    metricas_df[
                        "Esquina"
                    ]
                    == esquina
                )

                &

                (
                    metricas_df[
                        "Metrica_evaluada"
                    ]
                    == "Delay reconstruido"
                )
            )

            encontrados = (
                metricas_df.loc[
                    filtro
                ]
            )

            if len(encontrados):

                fila = encontrados.iloc[0]

                filas_tabla.append({

                    "Tipo": "Mejor modelo",

                    "Esquina": esquina,

                    "Objetivo": objetivo,

                    "Modelo": modelo,

                    "Conjunto": conjunto,

                    "MAE": fila["MAE"],

                    "RMSE": fila["RMSE"],

                    "R2": fila["R2"],

                    "Pearson": fila["Pearson"],
                })

    tabla_4_conjuntos = pd.DataFrame(
        filas_tabla
    )

    tabla_4_conjuntos.to_csv(

        RESULTS_DIR
        / "tabla_4_conjuntos.csv",

        index=False,
    )

    # ========================================================
    # COMPARACION DELTA VS DELAY DIRECTO
    # ========================================================

    comparacion_delta = []

    for esquina in (
        "Slow",
        "Fast",
    ):

        for conjunto in (
            "Validacion",
            "Test Labels",
            "Test Designs",
        ):

            # --------------------------------------------
            # Mejor Delay directo
            # --------------------------------------------

            filtro_delay = (

                (mejores_df["Esquina"] == esquina)

                &

                (
                    mejores_df["Objetivo"]
                    == "Delay"
                )
            )

            mejor_delay_df = (
                mejores_df.loc[
                    filtro_delay
                ]
            )

            # --------------------------------------------
            # Mejor Delta
            # --------------------------------------------

            filtro_delta = (

                (mejores_df["Esquina"] == esquina)

                &

                (
                    mejores_df["Objetivo"]
                    == "Delta"
                )
            )

            mejor_delta_df = (
                mejores_df.loc[
                    filtro_delta
                ]
            )

            if (
                len(mejor_delay_df)
                and len(mejor_delta_df)
            ):

                md = (
                    mejor_delay_df.iloc[0]
                )

                mdelta = (
                    mejor_delta_df.iloc[0]
                )

                # Delay directo
                filtro_md = (

                    (metricas_df["Modelo"]
                     == md["Modelo"])

                    &

                    (
                        metricas_df[
                            "Experimento"
                        ]
                        == md["Experimento"]
                    )

                    &

                    (
                        metricas_df[
                            "Objetivo"
                        ]
                        == "Delay (final)"
                    )

                    &

                    (
                        metricas_df[
                            "Conjunto"
                        ]
                        == conjunto
                    )

                    &

                    (
                        metricas_df[
                            "Esquina"
                        ]
                        == esquina
                    )

                    &

                    (
                        metricas_df[
                            "Metrica_evaluada"
                        ]
                        == "Delay reconstruido"
                    )
                )

                # Delta
                filtro_mdelta = (

                    (metricas_df["Modelo"]
                     == mdelta["Modelo"])

                    &

                    (
                        metricas_df[
                            "Experimento"
                        ]
                        == mdelta["Experimento"]
                    )

                    &

                    (
                        metricas_df[
                            "Objetivo"
                        ]
                        == "Delta (final)"
                    )

                    &

                    (
                        metricas_df[
                            "Conjunto"
                        ]
                        == conjunto
                    )

                    &

                    (
                        metricas_df[
                            "Esquina"
                        ]
                        == esquina
                    )

                    &

                    (
                        metricas_df[
                            "Metrica_evaluada"
                        ]
                        == "Delay reconstruido"
                    )
                )

                delay_rows = (
                    metricas_df.loc[
                        filtro_md
                    ]
                )

                delta_rows = (
                    metricas_df.loc[
                        filtro_mdelta
                    ]
                )

                if (
                    len(delay_rows)
                    and len(delta_rows)
                ):

                    rmse_delay = (
                        delay_rows.iloc[0]["RMSE"]
                    )

                    rmse_delta = (
                        delta_rows.iloc[0]["RMSE"]
                    )

                    mejora = (
                        (
                            rmse_delay
                            - rmse_delta
                        )
                        / rmse_delay
                        * 100
                    )

                    comparacion_delta.append({

                        "Esquina": esquina,

                        "Conjunto": conjunto,

                        "Modelo_Delay":
                            md["Modelo"],

                        "RMSE_Delay":
                            rmse_delay,

                        "Modelo_Delta":
                            mdelta["Modelo"],

                        "RMSE_Delta":
                            rmse_delta,

                        "Mejora_Delta_porcentaje":
                            mejora,

                        "Delta_mejor":
                            rmse_delta
                            < rmse_delay,
                    })

    comparacion_delta_df = pd.DataFrame(
        comparacion_delta
    )

    comparacion_delta_df.to_csv(

        RESULTS_DIR
        / "comparacion_delta_vs_delay.csv",

        index=False,
    )

    # ========================================================
    # RESUMEN PARA REUNION
    # ========================================================

    filas_resumen = []

    for _, mejor in (
        mejores_df.iterrows()
    ):

        esquina = mejor[
            "Esquina"
        ]

        objetivo = mejor[
            "Objetivo"
        ]

        modelo = mejor[
            "Modelo"
        ]

        experimento = mejor[
            "Experimento"
        ]

        def obtener_metricas(
            conjunto,
            objetivo_final=True,
        ):

            if objetivo_final:

                objetivo_busqueda = (
                    f"{objetivo} (final)"
                )

            else:

                objetivo_busqueda = objetivo

            filtro = (

                (metricas_df["Modelo"]
                 == modelo)

                &

                (
                    metricas_df[
                        "Experimento"
                    ]
                    == experimento
                )

                &

                (
                    metricas_df[
                        "Objetivo"
                    ]
                    == objetivo_busqueda
                )

                &

                (
                    metricas_df[
                        "Conjunto"
                    ]
                    == conjunto
                )

                &

                (
                    metricas_df[
                        "Esquina"
                    ]
                    == esquina
                )

                &

                (
                    metricas_df[
                        "Metrica_evaluada"
                    ]
                    == "Delay reconstruido"
                )
            )

            encontrados = (
                metricas_df.loc[
                    filtro
                ]
            )

            if not len(encontrados):
                return {
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "Pearson": np.nan,
                }

            fila = encontrados.iloc[0]

            return {

                "MAE": fila["MAE"],

                "RMSE": fila["RMSE"],

                "R2": fila["R2"],

                "Pearson": fila["Pearson"],
            }

        for conjunto in (
            "Train final",
            "Validacion",
            "Test Labels",
            "Test Designs",
        ):

            valores = obtener_metricas(
                conjunto
            )

            filas_resumen.append({

                "Esquina": esquina,

                "Objetivo": objetivo,

                "Modelo": modelo,

                "Experimento": experimento,

                "Conjunto": conjunto,

                "MAE": valores["MAE"],

                "RMSE": valores["RMSE"],

                "R2": valores["R2"],

                "Pearson": valores["Pearson"],
            })

    resumen_df = pd.DataFrame(
        filas_resumen
    )

    resumen_df.to_csv(

        RESULTS_DIR
        / "resumen_para_reunion.csv",

        index=False,
    )

    # ========================================================
    # IMPRIMIR RESULTADOS IMPORTANTES
    # ========================================================

    print("\n" + "=" * 70)
    print("MEJORES MODELOS")
    print("=" * 70)

    print(
        mejores_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}"
        )
    )

    print("\n" + "=" * 70)
    print("TABLA DE 4 CONJUNTOS")
    print("=" * 70)

    print(
        tabla_4_conjuntos.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}"
        )
    )

    print("\n" + "=" * 70)
    print("COMPARACION DELTA VS DELAY")
    print("=" * 70)

    if len(comparacion_delta_df):

        print(
            comparacion_delta_df.to_string(
                index=False,
                float_format=lambda x:
                f"{x:.6f}"
            )
        )

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    if len(importancias_df):

        # Mostrar solamente RF mejor.
        rf_importancias = (
            importancias_df[
                importancias_df[
                    "Modelo"
                ]
                == "Random Forest - mejor"
            ]
            .sort_values(
                [
                    "Esquina",
                    "Objetivo",
                    "Importancia",
                ],
                ascending=[
                    True,
                    True,
                    False,
                ],
            )
        )

        print(
            rf_importancias.to_string(
                index=False,
                float_format=lambda x:
                f"{x:.6f}"
            )
        )

    # ========================================================
    # UBICACION DE RESULTADOS
    # ========================================================

    print("\n" + "=" * 70)
    print("ARCHIVOS GENERADOS")
    print("=" * 70)

    print(
        f"\nResultados:"
        f"\n{RESULTS_DIR}"
    )

    print(
        "\nArchivos principales:"
    )

    print(
        " - metricas_objetivo3.csv"
    )

    print(
        " - mejores_modelos.csv"
    )

    print(
        " - tabla_4_conjuntos.csv"
    )

    print(
        " - resumen_para_reunion.csv"
    )

    print(
        " - comparacion_delta_vs_delay.csv"
    )

    print(
        " - diagnostico_distribucion_delta.csv"
    )

    print(
        " - importancia_variables.csv"
    )

    print(
        " - predicciones_test_labels.csv"
    )

    print(
        " - predicciones_test_designs.csv"
    )

    print(
        " - ranking_slow_delay.csv"
    )

    print(
        " - ranking_fast_delay.csv"
    )

    print(
        " - ranking_slow_delta.csv"
    )

    print(
        " - ranking_fast_delta.csv"
    )

    print("\n" + "=" * 70)
    print("PROCESO TERMINADO")
    print("=" * 70)


# ============================================================
# EJECUCION
# ============================================================

if __name__ == "__main__":
    main()