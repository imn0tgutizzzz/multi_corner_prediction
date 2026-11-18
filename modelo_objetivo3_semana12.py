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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor



from hybridmodel_min_samples_leaf import HybridModel

# Experimento del modelo híbrido: minimo de muestras por hoja.
# Valores previstos para comparar: 1, 2, 4, 8 y 16.
# La ejecucion actual usa HYBRID_MIN_SAMPLES_LEAF; para cambiar de experimento,
# seleccionar uno de esos valores. El modelo conserva los demas parametros.
HYBRID_MIN_SAMPLES_LEAF = 4
HYBRID_MIN_SAMPLES_LEAF_EXPERIMENTS = (1, 2, 4, 8, 16)


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
    # GRADIENT BOOSTING - BARRIDO CONTROLADO DE HOJAS
    # Solo cambia min_samples_leaf; los demas parametros quedan fijos.
    # --------------------------------------------------------

    for hojas in (1, 2, 4, 8, 16):
        experimentos.append(
            (
                "Gradient Boosting",
                {
                    "n_estimators": 100,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "min_samples_leaf": hojas,
                }
            )
        )

    # --------------------------------------------------------
    # HIST GRADIENT BOOSTING - BARRIDO CONTROLADO DE HOJAS
    # --------------------------------------------------------

    for hojas in (1, 2, 4, 8, 16):
        experimentos.append(
            (
                "Hist Gradient Boosting",
                {
                    "max_iter": 200,
                    "learning_rate": 0.05,
                    "max_leaf_nodes": 31,
                    "min_samples_leaf": hojas,
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
    # RANDOM FOREST - MIN SAMPLES LEAF Y CANTIDAD DE ARBOLES
    # Barrido combinado: para cada min_samples_leaf se prueban
    # todos los valores de n_estimators.
    #
    # Se mantienen max_depth=20 y los demas parametros fijos.
    # Esto permite observar si el efecto del tamano minimo de hoja
    # cambia al variar la cantidad de arboles.
    # --------------------------------------------------------

    for hoja in (1, 2, 4, 8, 16):
        for n in (10, 50, 100, 200, 300, 500):
            experimentos.append(
                (
                    "Random Forest",
                    {
                        "n_estimators": n,
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
# DIAGNOSTICO DE ERRORES PARA EL BARRIDO DE MIN_SAMPLES_LEAF
# ============================================================

def guardar_histograma_errores(y_real, y_pred, titulo, ruta):
    """Guarda la distribucion del error firmado y del error absoluto."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errores = y_pred - y_real
    errores = errores[np.isfinite(errores)]
    if not len(errores):
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(errores, bins=80, alpha=0.8, edgecolor='black', linewidth=0.25)
    ax.axvline(0, linestyle='--', color='black', linewidth=1.2)
    ax.set_title(titulo)
    ax.set_xlabel('Error firmado = prediccion - valor real')
    ax.set_ylabel('Numero de muestras')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(ruta, dpi=180)
    plt.close(fig)


def guardar_errores_por_diseno(datos, y_real, y_pred, conjunto, modelo, hojas):
    """Guarda errores por muestra y un resumen agregado por diseño."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_real
    error_abs = np.abs(error)

    detalle = pd.DataFrame({
        'Conjunto': conjunto,
        'Modelo': modelo,
        'min_samples_leaf': hojas,
        'Delta_Fast_real': y_real,
        'Delta_Fast_predicho': y_pred,
        'Error_firmado': error,
        'Error_absoluto': error_abs,
    })
    # Incluye una clave de diseño cuando existe; si no, usa la descripción.
    if 'Design' in datos.columns:
        detalle.insert(0, 'Design', datos['Design'].astype(str).to_numpy())
        clave = 'Design'
    elif 'Description' in datos.columns:
        detalle.insert(0, 'Description', datos['Description'].astype(str).to_numpy())
        clave = 'Description'
    else:
        detalle.insert(0, 'Fila', np.arange(len(datos)))
        clave = 'Fila'

    leaf_text = str(hojas).zfill(2)
    safe_model = modelo.lower().replace(' ', '_')
    detalle_path = RESULTS_DIR / f'errores_fast_delta_{conjunto.lower().replace(" ", "_")}_{safe_model}_leaf_{leaf_text}.csv'
    detalle.to_csv(detalle_path, index=False)

    resumen = detalle.groupby(clave, dropna=False).agg(
        N=('Error_absoluto', 'size'),
        MAE=('Error_absoluto', 'mean'),
        RMSE=('Error_firmado', lambda x: float(np.sqrt(np.mean(np.square(x))))),
        Error_absoluto_max=('Error_absoluto', 'max'),
        Error_firmado_medio=('Error_firmado', 'mean'),
    ).reset_index()
    resumen.insert(0, 'Conjunto', conjunto)
    resumen.insert(1, 'Modelo', modelo)
    resumen.insert(2, 'min_samples_leaf', hojas)
    resumen = resumen.sort_values(['RMSE', 'Error_absoluto_max'], ascending=False)
    resumen_path = RESULTS_DIR / f'errores_por_diseno_fast_delta_{conjunto.lower().replace(" ", "_")}_{safe_model}_leaf_{leaf_text}.csv'
    resumen.to_csv(resumen_path, index=False)

    # Histograma con todos los errores, sin eliminar los extremos.
    guardar_histograma_errores(
        y_real, y_pred,
        f'{conjunto} - Fast Delta - {modelo} - min_samples_leaf={hojas}',
        GRAFICAS_DIR / f'hist_error_fast_delta_{conjunto.lower().replace(" ", "_")}_{safe_model}_leaf_{leaf_text}.png',
    )

    return detalle, resumen


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

# ============================================================
# EXPERIMENTO ADICIONAL: CLASIFICADOR + REGRESOR (DELTA)
# ============================================================
# Se agrega este experimento sin reemplazar los experimentos anteriores.
#
# Etapa 1: clasificar si |Delta| <= tolerancia (delta aproximadamente cero).
# Etapa 2: si no es aproximadamente cero, estimar Delta con un regresor
# entrenado únicamente con los ejemplos cuyo |Delta| > tolerancia.
#
# Si el clasificador predice Delta aproximadamente cero, se asigna Delta=0,
# es decir, se usa Delay_Typical directamente.
# ============================================================


def ejecutar_experimento_clasificador_regresor(
    train,
    conjuntos_test,
    columnas_features,
    indices_train,
    indices_validacion,
    tolerancia=0.05,
    metricas_df=None,
):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENTO ADICIONAL: CLASIFICADOR + REGRESOR")
    print("=" * 70)
    print(
        "Tolerancia para considerar Delta aproximadamente cero:",
        tolerancia,
    )

    X_completo = preparar_X(
        train,
        columnas_features,
    )

    X_train_experimento = X_completo.iloc[
        indices_train
    ]

    filas_resultados = []
    predicciones_guardadas = []
    filas_metricas_clasificador = []
    filas_comparacion_modelos = []

    for esquina in ("Slow", "Fast"):

        columna_delta = f"Delta_{esquina}"
        columna_delay = f"Delay_{esquina}"

        delta_train = train[
            columna_delta
        ].iloc[indices_train].astype(float)

        mascara_valida = np.isfinite(
            delta_train.to_numpy()
        )

        X_train_valido = X_train_experimento.loc[
            mascara_valida
        ]

        delta_train_valido = delta_train.loc[
            mascara_valida
        ]

        # Etiqueta 1: delta dentro de la tolerancia.
        # Etiqueta 0: delta que requiere regresión.
        y_clasificador = (
            delta_train_valido.abs() <= tolerancia
        ).astype(int)

        if y_clasificador.nunique() < 2:
            print(
                f"\n{esquina}: no se ejecuta el experimento porque "
                "el conjunto de entrenamiento solo contiene una clase."
            )
            continue

        clasificador = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("modelo", RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

        clasificador.fit(
            X_train_valido,
            y_clasificador,
        )

        # El regresor se entrena solo con deltas fuera de la tolerancia.
        mascara_regresion = (
            delta_train_valido.abs() > tolerancia
        )

        X_regresion = X_train_valido.loc[
            mascara_regresion
        ]

        y_regresion = delta_train_valido.loc[
            mascara_regresion
        ]

        if len(y_regresion) == 0:
            print(
                f"\n{esquina}: no se ejecuta la regresión porque "
                "no hay ejemplos fuera de la tolerancia."
            )
            continue

        regresor = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("modelo", RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

        regresor.fit(
            X_regresion,
            y_regresion,
        )

        print("\n" + "-" * 70)
        print(f"ESQUINA: {esquina}")
        print(
            f"Ejemplos usados por el clasificador: "
            f"{len(y_clasificador):,}"
        )
        print(
            f"Ejemplos usados por el regresor: "
            f"{len(y_regresion):,}"
        )
        print(
            f"Ejemplos con |Delta| <= {tolerancia}: "
            f"{int(y_clasificador.sum()):,}"
        )

        conjuntos_evaluacion = {
            "Validacion": train.iloc[
                indices_validacion
            ],
            **conjuntos_test,
        }

        for nombre_conjunto, datos in conjuntos_evaluacion.items():

            X_eval = preparar_X(
                datos,
                columnas_features,
            )

            delta_real = datos[
                columna_delta
            ].astype(float).to_numpy()

            delay_typical = datos[
                "Delay_Typical"
            ].astype(float).to_numpy()

            delay_real = datos[
                columna_delay
            ].astype(float).to_numpy()

            validos = (
                np.isfinite(delta_real)
                & np.isfinite(delay_typical)
                & np.isfinite(delay_real)
            )

            if not validos.any():
                continue

            X_eval_valido = X_eval.loc[validos]
            delta_real_valido = delta_real[validos]
            delay_typical_valido = delay_typical[validos]
            delay_real_valido = delay_real[validos]

            clase_real = (
                np.abs(delta_real_valido) <= tolerancia
            ).astype(int)

            clase_predicha = clasificador.predict(
                X_eval_valido
            )

            # Evaluación del clasificador:
            # clase 1 = Delta casi cero / usar Typical.
            # clase 0 = Delta significativo / usar regresor.
            tn, fp, fn, tp = confusion_matrix(
                clase_real,
                clase_predicha,
                labels=[0, 1],
            ).ravel()

            precision_clases, recall_clases, f1_clases, _ = (
                precision_recall_fscore_support(
                    clase_real,
                    clase_predicha,
                    labels=[0, 1],
                    zero_division=0,
                )
            )

            filas_metricas_clasificador.append({
                "Esquina": esquina,
                "Conjunto": nombre_conjunto,
                "Tolerancia": tolerancia,
                "N": int(len(clase_real)),
                "N_real_delta_cero": int((clase_real == 1).sum()),
                "N_real_regresion": int((clase_real == 0).sum()),
                "TN_regresion_correcta": int(tn),
                "FP_regresion_enviado_a_typical": int(fp),
                "FN_delta_cero_enviado_a_regresor": int(fn),
                "TP_delta_cero_correcto": int(tp),
                "Accuracy": float(accuracy_score(
                    clase_real,
                    clase_predicha,
                )),
                "Balanced_Accuracy": float(balanced_accuracy_score(
                    clase_real,
                    clase_predicha,
                )),
                "Precision_clase_regresion": float(precision_clases[0]),
                "Recall_clase_regresion": float(recall_clases[0]),
                "F1_clase_regresion": float(f1_clases[0]),
                "Precision_clase_delta_cero": float(precision_clases[1]),
                "Recall_clase_delta_cero": float(recall_clases[1]),
                "F1_clase_delta_cero": float(f1_clases[1]),
            })

            delta_predicho = np.zeros(
                len(X_eval_valido),
                dtype=float,
            )

            # La clase 0 indica que el delta no es aproximadamente cero.
            requiere_regresion = (
                clase_predicha == 0
            )

            if requiere_regresion.any():
                delta_predicho[requiere_regresion] = (
                    regresor.predict(
                        X_eval_valido.loc[
                            requiere_regresion
                        ]
                    )
                )

            delay_predicho = (
                delay_typical_valido
                + delta_predicho
            )

            metricas_delta = metricas(
                delta_real_valido,
                delta_predicho,
            )

            metricas_delay = metricas(
                delay_real_valido,
                delay_predicho,
            )

            filas_resultados.append({
                "Esquina": esquina,
                "Conjunto": nombre_conjunto,
                "Tolerancia": tolerancia,
                "N": int(validos.sum()),
                "N_predicho_delta_cero": int(
                    (clase_predicha == 1).sum()
                ),
                "N_predicho_regresion": int(
                    requiere_regresion.sum()
                ),
                "MAE_Delta": metricas_delta["MAE"],
                "RMSE_Delta": metricas_delta["RMSE"],
                "R2_Delta": metricas_delta["R2"],
                "Pearson_Delta": metricas_delta["Pearson"],
                "MAE_Delay": metricas_delay["MAE"],
                "RMSE_Delay": metricas_delay["RMSE"],
                "R2_Delay": metricas_delay["R2"],
                "Pearson_Delay": metricas_delay["Pearson"],
            })

            predicciones_guardadas.append(pd.DataFrame({
                "Esquina": esquina,
                "Conjunto": nombre_conjunto,
                "Delay_Typical": delay_typical_valido,
                "Delta_real": delta_real_valido,
                "Delta_predicho": delta_predicho,
                "Delay_real": delay_real_valido,
                "Delay_predicho": delay_predicho,
                "Clase_delta_cero_real": clase_real,
                "Clase_delta_cero_predicha": clase_predicha,
                "Usa_regresor": requiere_regresion,
            }))

            # Comparación directa de métricas: baseline, modelo anterior
            # (objetivo "Delta (final)") y el experimento nuevo.
            baseline_delta_metrics = metricas(
                delta_real_valido,
                np.zeros(len(delta_real_valido)),
            )
            baseline_delay_metrics = metricas(
                delay_real_valido,
                delay_typical_valido,
            )

            modelos_comparar = [
                {
                    "Modelo": "Baseline Typical",
                    "MAE_Delta": baseline_delta_metrics["MAE"],
                    "RMSE_Delta": baseline_delta_metrics["RMSE"],
                    "R2_Delta": baseline_delta_metrics["R2"],
                    "Pearson_Delta": baseline_delta_metrics["Pearson"],
                    "MAE_Delay": baseline_delay_metrics["MAE"],
                    "RMSE_Delay": baseline_delay_metrics["RMSE"],
                    "R2_Delay": baseline_delay_metrics["R2"],
                    "Pearson_Delay": baseline_delay_metrics["Pearson"],
                },
                {
                    "Modelo": "Clasificador + Regresor",
                    "MAE_Delta": metricas_delta["MAE"],
                    "RMSE_Delta": metricas_delta["RMSE"],
                    "R2_Delta": metricas_delta["R2"],
                    "Pearson_Delta": metricas_delta["Pearson"],
                    "MAE_Delay": metricas_delay["MAE"],
                    "RMSE_Delay": metricas_delay["RMSE"],
                    "R2_Delay": metricas_delay["R2"],
                    "Pearson_Delay": metricas_delay["Pearson"],
                },
            ]

            if metricas_df is not None and not metricas_df.empty:
                filtro_previo = (
                    (metricas_df["Objetivo"] == "Delta (final)")
                    & (metricas_df["Esquina"] == esquina)
                    & (metricas_df["Conjunto"] == nombre_conjunto)
                )

                fila_previa_delta = metricas_df[
                    filtro_previo
                    & (metricas_df["Metrica_evaluada"] == "Delta")
                ]

                fila_previa_delay = metricas_df[
                    filtro_previo
                    & (
                        metricas_df["Metrica_evaluada"]
                        == "Delay reconstruido"
                    )
                ]

                if (
                    not fila_previa_delta.empty
                    and not fila_previa_delay.empty
                ):
                    fila_previa_delta = fila_previa_delta.iloc[0]
                    fila_previa_delay = fila_previa_delay.iloc[0]

                    modelos_comparar.insert(
                        1,
                        {
                            "Modelo": (
                                "Modelo anterior: "
                                f"{fila_previa_delta.get('Modelo', 'Delta final')}"
                            ),
                            "MAE_Delta": fila_previa_delta["MAE"],
                            "RMSE_Delta": fila_previa_delta["RMSE"],
                            "R2_Delta": fila_previa_delta["R2"],
                            "Pearson_Delta": fila_previa_delta["Pearson"],
                            "MAE_Delay": fila_previa_delay["MAE"],
                            "RMSE_Delay": fila_previa_delay["RMSE"],
                            "R2_Delay": fila_previa_delay["R2"],
                            "Pearson_Delay": fila_previa_delay["Pearson"],
                        },
                    )

            for fila_comp in modelos_comparar:
                fila_comp.update({
                    "Esquina": esquina,
                    "Conjunto": nombre_conjunto,
                    "Tolerancia": tolerancia,
                    "N": int(validos.sum()),
                })
                filas_comparacion_modelos.append(fila_comp)

    resultados_df = pd.DataFrame(
        filas_resultados
    )

    if len(resultados_df):
        resultados_df.to_csv(
            RESULTS_DIR
            / "resultados_clasificador_regresor_delta.csv",
            index=False,
        )

    if len(predicciones_guardadas):
        pd.concat(
            predicciones_guardadas,
            ignore_index=True,
        ).to_csv(
            RESULTS_DIR
            / "predicciones_clasificador_regresor_delta.csv",
            index=False,
        )

    metricas_clasificador_df = pd.DataFrame(
        filas_metricas_clasificador
    )

    if not metricas_clasificador_df.empty:
        metricas_clasificador_df.to_csv(
            RESULTS_DIR
            / "metricas_clasificador_delta_cero.csv",
            index=False,
        )

    comparacion_modelos_df = pd.DataFrame(
        filas_comparacion_modelos
    )

    if not comparacion_modelos_df.empty:
        comparacion_modelos_df.to_csv(
            RESULTS_DIR
            / "comparacion_clasificador_regresor_baseline_modelo_anterior.csv",
            index=False,
        )

    print("\nResultados del experimento clasificador + regresor:")

    if len(resultados_df):
        print(
            resultados_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )
        print(
            "\nCSV generado: "
            "resultados_clasificador_regresor_delta.csv"
        )
        print(
            "Predicciones guardadas en: "
            "predicciones_clasificador_regresor_delta.csv"
        )
    else:
        print(
            "No se generaron resultados; revise la distribución de Delta."
        )

    print("\nMétricas del clasificador:")
    if not metricas_clasificador_df.empty:
        print(
            metricas_clasificador_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )
        print(
            "CSV generado: metricas_clasificador_delta_cero.csv"
        )

    print("\nComparación contra baseline y modelo anterior:")
    if not comparacion_modelos_df.empty:
        print(
            comparacion_modelos_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )
        print(
            "CSV generado: "
            "comparacion_clasificador_regresor_baseline_modelo_anterior.csv"
        )


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

    # Factor de mezcla elegido solo con validacion.
    # alpha=0 equivale al baseline Typical; alpha=1 usa el modelo completo.
    alphas_baseline_aware = {}

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
                # NUEVO: diagnostico dedicado para Fast Delta
                # en las configuraciones del barrido de hojas.
                # --------------------------------------------
                if (
                    esquina == "Fast"
                    and objetivo == "Delta"
                    and nombre_modelo in (
                        "Gradient Boosting",
                        "Hist Gradient Boosting",
                    )
                ):
                    # Registrar la configuracion experimental en VALIDACION
                    # y en los dos conjuntos de prueba. Los conjuntos de prueba
                    # se usan solo para evaluar, nunca para elegir el ganador.
                    hojas = int(parametros["min_samples_leaf"])

                    registrar(
                        resultados,
                        nombre_modelo,
                        experimento,
                        "Fast Delta (barrido)",
                        "Validacion",
                        "Fast",
                        train["Delta_Fast"].iloc[indices_validacion],
                        pred_target_validacion,
                        len(X_ajuste),
                        "Delta",
                    )

                    for conjunto_diag, datos_diag in conjuntos_test.items():
                        pred_delta_diag = predecir_target(
                            modelo,
                            X_tests[conjunto_diag]
                        )

                        registrar(
                            resultados,
                            nombre_modelo,
                            experimento,
                            "Fast Delta (barrido)",
                            conjunto_diag,
                            "Fast",
                            datos_diag["Delta_Fast"].to_numpy(float),
                            pred_delta_diag,
                            len(X_ajuste),
                            "Delta",
                        )

                        guardar_errores_por_diseno(
                            datos_diag,
                            datos_diag["Delta_Fast"].to_numpy(float),
                            pred_delta_diag,
                            conjunto_diag,
                            nombre_modelo,
                            hojas,
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

            # =================================================
            # CALIBRACION BASELINE-AWARE (SOLO VALIDACION)
            # =================================================
            # No se utilizan Test Labels ni Test Designs para elegir alpha.
            # alpha=0 -> prediccion Typical; alpha=1 -> prediccion ML completa.
            modelo_validacion = mejor["Modelo_entrenado"]
            pred_validacion_modelo = predecir_delay(
                modelo_validacion,
                X_validacion,
                objetivo
            )
            baseline_validacion = train[
                "Delay_Typical"
            ].iloc[indices_validacion].to_numpy(dtype=float)
            real_validacion_delay = train[
                f"Delay_{esquina}"
            ].iloc[indices_validacion].to_numpy(dtype=float)

            alphas_candidatos = np.linspace(0.0, 1.0, 101)
            rmse_por_alpha = []
            for alpha_candidato in alphas_candidatos:
                pred_mezcla = (
                    baseline_validacion
                    + alpha_candidato
                    * (np.asarray(pred_validacion_modelo, dtype=float)
                       - baseline_validacion)
                )
                rmse_alpha = float(np.sqrt(np.mean(
                    (real_validacion_delay - pred_mezcla) ** 2
                )))
                rmse_por_alpha.append(rmse_alpha)

            alpha_elegido = float(
                alphas_candidatos[int(np.argmin(rmse_por_alpha))]
            )
            alphas_baseline_aware[(esquina, objetivo)] = alpha_elegido

            # Registrar la mezcla en validacion para poder compararla
            # directamente con baseline y modelo original.
            pred_validacion_mezcla = (
                baseline_validacion
                + alpha_elegido
                * (np.asarray(pred_validacion_modelo, dtype=float)
                   - baseline_validacion)
            )
            registrar(
                resultados,
                "Baseline-aware blend",
                f"alpha={alpha_elegido:.2f}; base={mejor_nombre}",
                f"{objetivo} (calibrado)",
                "Validacion",
                esquina,
                real_validacion_delay,
                pred_validacion_mezcla,
                len(X_train),
                "Delay reconstruido",
            )

            print(
                f"Alpha baseline-aware para {esquina} {objetivo}: "
                f"{alpha_elegido:.2f} "
                f"(0=baseline, 1=modelo completo)"
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
                # NUEVO: mezcla calibrada contra baseline
                # --------------------------------------------
                # El alpha se elige con validacion; nunca con el conjunto
                # de prueba. La mezcla se acerca a Typical si el modelo
                # completo no aporta suficiente mejora en validacion.
                alpha_blend = alphas_baseline_aware[(esquina, objetivo)]
                baseline_delay = datos["Delay_Typical"].to_numpy(dtype=float)
                pred_delay_blend = (
                    baseline_delay
                    + alpha_blend
                    * (np.asarray(pred_delay, dtype=float) - baseline_delay)
                )

                registrar(
                    resultados,
                    "Baseline-aware blend",
                    f"alpha={alpha_blend:.2f}; base={mejor_nombre}",
                    f"{objetivo} (calibrado)",
                    conjunto,
                    esquina,
                    datos[f"Delay_{esquina}"],
                    pred_delay_blend,
                    len(X_final),
                    "Delay reconstruido",
                )

                # --------------------------------------------
                # Guardar predicciones
                # --------------------------------------------

                predicciones[
                    (conjunto, esquina, objetivo)
                ] = {
                    "target": pred_target,
                    "delay": pred_delay,
                    }

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
    # MODELO HIBRIDO: FAST DELTA
    # Se evalúa aparte para compararlo con el baseline y Random Forest.
    # No cambia la selección de los ganadores de los otros experimentos.
    # ========================================================

    print("\n" + "=" * 70)
    print("MODELO HIBRIDO - FAST DELTA")
    print("=" * 70)

    # Se utiliza la clase HybridModel ya existente en hybridmodel.py.
    # El archivo hybridmodel.py no se modifica.
    # Como esa implementación no trata valores NaN, se imputan las features
    # aquí, dentro de objetivo3, sin alterar el procesamiento de los demás modelos.
    # La imputación de validación se ajusta solo con el 80 % de entrenamiento.
    imputer_hybrid_validacion = SimpleImputer(strategy="median")
    X_train_hybrid = pd.DataFrame(
        imputer_hybrid_validacion.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_validacion_hybrid = pd.DataFrame(
        imputer_hybrid_validacion.transform(X_validacion),
        columns=X_validacion.columns,
        index=X_validacion.index,
    )

    hybrid_validacion = HybridModel(min_samples_leaf=HYBRID_MIN_SAMPLES_LEAF)
    hybrid_validacion.fit(
        X_train_hybrid,
        train["Delta_Fast"].iloc[indices_train],
        LR_type=0,
        custom_params=None,
    )

    pred_hybrid_validacion_delta = np.asarray(
        hybrid_validacion.predict(X_validacion_hybrid), dtype=float
    ).reshape(-1)
    pred_hybrid_validacion_delay = (
        train["Delay_Typical"].iloc[indices_validacion].to_numpy(dtype=float)
        + pred_hybrid_validacion_delta
    )

    # Random Forest de referencia: configuración utilizada en los avances previos.
    parametros_rf_referencia = {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 4,
    }
    rf_validacion = crear_modelo("Random Forest", parametros_rf_referencia)
    rf_validacion.fit(
        X_train,
        train["Delta_Fast"].iloc[indices_train],
    )
    pred_rf_validacion_delta = np.asarray(
        rf_validacion.predict(X_validacion), dtype=float
    ).reshape(-1)
    pred_rf_validacion_delay = (
        train["Delay_Typical"].iloc[indices_validacion].to_numpy(dtype=float)
        + pred_rf_validacion_delta
    )

    # Entrenamiento final con todas las filas de Train.
    imputer_hybrid_final = SimpleImputer(strategy="median")
    X_hybrid_final = pd.DataFrame(
        imputer_hybrid_final.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )

    hybrid_final = HybridModel(min_samples_leaf=HYBRID_MIN_SAMPLES_LEAF)
    hybrid_final.fit(
        X_hybrid_final,
        train["Delta_Fast"],
        LR_type=0,
        custom_params=None,
    )
    # Se guardan juntos el imputador y el modelo para reproducir las predicciones.
    joblib.dump(
        {
            "imputer": imputer_hybrid_final,
            "model": hybrid_final,
            "features": list(X.columns),
        },
        RESULTS_DIR / "modelo_hibrido_fast_delta.joblib",
    )

    rf_final_fast_delta = crear_modelo("Random Forest", parametros_rf_referencia)
    rf_final_fast_delta.fit(X, train["Delta_Fast"])

    evaluaciones_hibrido = []
    datos_validacion = train.iloc[indices_validacion]

    # Validación: modelos entrenados con el 80 % de Train.
    evaluaciones_hibrido.append((
        "Validacion", datos_validacion,
        pred_hybrid_validacion_delta, pred_hybrid_validacion_delay,
        pred_rf_validacion_delta, pred_rf_validacion_delay,
    ))

    # Test Labels y Test Designs: modelos finales entrenados con todo Train.
    for nombre_conjunto, datos_conjunto in conjuntos_test.items():
        X_conjunto = X_tests[nombre_conjunto]
        X_conjunto_hybrid = pd.DataFrame(
            imputer_hybrid_final.transform(X_conjunto),
            columns=X_conjunto.columns,
            index=X_conjunto.index,
        )
        pred_h_delta = np.asarray(
            hybrid_final.predict(X_conjunto_hybrid), dtype=float
        ).reshape(-1)
        pred_h_delay = (
            datos_conjunto["Delay_Typical"].to_numpy(dtype=float) + pred_h_delta
        )
        pred_rf_delta = np.asarray(
            rf_final_fast_delta.predict(X_conjunto), dtype=float
        ).reshape(-1)
        pred_rf_delay = (
            datos_conjunto["Delay_Typical"].to_numpy(dtype=float) + pred_rf_delta
        )
        evaluaciones_hibrido.append((
            nombre_conjunto, datos_conjunto,
            pred_h_delta, pred_h_delay,
            pred_rf_delta, pred_rf_delay,
        ))

    resumen_comparacion_hibrido = []
    for (nombre_conjunto, datos_conjunto, pred_h_delta, pred_h_delay,
         pred_rf_delta, pred_rf_delay) in evaluaciones_hibrido:

        delta_real = datos_conjunto["Delta_Fast"].to_numpy(dtype=float)
        delay_real = datos_conjunto["Delay_Fast"].to_numpy(dtype=float)
        delay_typical = datos_conjunto["Delay_Typical"].to_numpy(dtype=float)
        pred_baseline_delta = np.zeros(len(datos_conjunto), dtype=float)
        pred_baseline_delay = delay_typical

        predicciones_comparar = [
            ("Baseline", pred_baseline_delta, pred_baseline_delay),
            ("Random Forest", pred_rf_delta, pred_rf_delay),
            ("Hybrid Model", pred_h_delta, pred_h_delay),
        ]

        for nombre_modelo_comp, pred_delta_comp, pred_delay_comp in predicciones_comparar:
            met_delta = metricas(delta_real, pred_delta_comp)
            met_delay = metricas(delay_real, pred_delay_comp)
            registrar(
                resultados,
                nombre_modelo_comp,
                "Fast Delta comparación",
                "Fast Delta (comparacion)",
                nombre_conjunto,
                "Fast",
                delta_real,
                pred_delta_comp,
                len(X_train) if nombre_conjunto == "Validacion" else len(X),
                "Delta",
            )
            registrar(
                resultados,
                nombre_modelo_comp,
                "Fast Delta comparación",
                "Fast Delta (comparacion)",
                nombre_conjunto,
                "Fast",
                delay_real,
                pred_delay_comp,
                len(X_train) if nombre_conjunto == "Validacion" else len(X),
                "Delay reconstruido",
            )

            error_abs = np.abs(pred_delta_comp - delta_real)
            resumen_comparacion_hibrido.append({
                "Conjunto": nombre_conjunto,
                "Modelo": nombre_modelo_comp,
                "MAE_Delta": met_delta["MAE"],
                "RMSE_Delta": met_delta["RMSE"],
                "R2_Delta": met_delta["R2"],
                "Pearson_Delta": met_delta["Pearson"],
                "MAE_Delay": met_delay["MAE"],
                "RMSE_Delay": met_delay["RMSE"],
                "R2_Delay": met_delay["R2"],
                "Pearson_Delay": met_delay["Pearson"],
                "Error_absoluto_P95_Delta": float(np.percentile(error_abs, 95)),
                "Error_absoluto_P99_Delta": float(np.percentile(error_abs, 99)),
                "Error_absoluto_max_Delta": float(np.max(error_abs)),
                "N_error_abs_mayor_0_1": int(np.sum(error_abs > 0.1)),
                "N_error_abs_mayor_0_5": int(np.sum(error_abs > 0.5)),
                "N_error_abs_mayor_1": int(np.sum(error_abs > 1.0)),
            })

            # Detalle de errores para localizar valores extremos.
            detalle_errores = datos_conjunto[COLUMNAS_LLAVE].copy()
            detalle_errores["Delta_Fast_real"] = delta_real
            detalle_errores["Delta_Fast_predicho"] = pred_delta_comp
            detalle_errores["Error_firmado"] = pred_delta_comp - delta_real
            detalle_errores["Error_absoluto"] = error_abs
            nombre_seguro = nombre_modelo_comp.lower().replace(" ", "_")
            detalle_errores.to_csv(
                RESULTS_DIR / (
                    f"errores_fast_delta_hibrido_comparacion_"
                    f"{nombre_conjunto.lower().replace(' ', '_')}_{nombre_seguro}.csv"
                ),
                index=False,
            )

    pd.DataFrame(resumen_comparacion_hibrido).to_csv(
        RESULTS_DIR / "comparacion_hybrid_rf_baseline_fast_delta.csv",
        index=False,
    )

    print("\nComparación Fast Delta guardada en:")
    print(RESULTS_DIR / "comparacion_hybrid_rf_baseline_fast_delta.csv")

    # ========================================================
    # DATAFRAME GENERAL DE METRICAS
    # ========================================================

    metricas_df = pd.DataFrame(
        resultados
    )

    # ========================================================
    # EXPERIMENTOS RANDOM FOREST
    # Exporta resultados medidos del RF (incluido el barrido de hojas)
    # a un archivo dedicado. Se conservan los registros de metricas_df.
    # ========================================================
    experimentos_random_forest_df = metricas_df[
        metricas_df["Modelo"] == "Random Forest"
    ].copy()

    if not experimentos_random_forest_df.empty:
        experimentos_random_forest_df.to_csv(
            RESULTS_DIR / "experimentos_random_forest.csv",
            index=False,
        )
    else:
        # Crear el archivo incluso si, por alguna razón, no hay filas RF.
        pd.DataFrame(
            columns=metricas_df.columns
        ).to_csv(
            RESULTS_DIR / "experimentos_random_forest.csv",
            index=False,
        )

    # ========================================================
    # NUEVO: TABLA COMPARATIVA DEL BARRIDO DE MIN_SAMPLES_LEAF
    # Se muestran métricas de Fast Delta y del delay reconstruido.
    # ========================================================
    filtro_barrido = (
        metricas_df["Modelo"].isin([
            "Gradient Boosting",
            "Hist Gradient Boosting",
        ])
        & metricas_df["Experimento"].astype(str).str.contains("min_samples_leaf")
        & (metricas_df["Esquina"] == "Fast")
        & (metricas_df["Objetivo"] == "Delta")
        & metricas_df["Conjunto"].isin([
            "Validacion", "Test Labels", "Test Designs"
        ])
    )
    barrido_df = metricas_df.loc[filtro_barrido].copy()
    if not barrido_df.empty:
        barrido_df.to_csv(
            RESULTS_DIR / "comparacion_min_samples_leaf_fast_delta.csv",
            index=False,
        )

    # Tabla compacta para comparar directamente el RMSE de Fast Delta.
    barrido_delta = barrido_df[
        barrido_df["Metrica_evaluada"] == "Delta"
    ].copy()
    if not barrido_delta.empty:
        filtro_baseline_fast = (
            (metricas_df["Modelo"] == "Baseline")
            & (metricas_df["Objetivo"] == "Delta")
            & (metricas_df["Esquina"] == "Fast")
            & (metricas_df["Metrica_evaluada"] == "Delta")
            & metricas_df["Conjunto"].isin([
                "Validacion", "Test Labels", "Test Designs"
            ])
        )
        baseline_fast = metricas_df.loc[
            filtro_baseline_fast,
            ["Conjunto", "RMSE"]
        ].rename(columns={"RMSE": "RMSE_Delta_baseline"})
        barrido_delta = barrido_delta.merge(
            baseline_fast,
            on="Conjunto",
            how="left",
        )
        barrido_delta["Mejora_RMSE_vs_baseline_pct"] = np.where(
            barrido_delta["RMSE_Delta_baseline"] > 0,
            (barrido_delta["RMSE_Delta_baseline"] - barrido_delta["RMSE"])
            / barrido_delta["RMSE_Delta_baseline"] * 100,
            np.nan,
        )
        barrido_delta["Mejora_vs_baseline"] = (
            barrido_delta["RMSE"] < barrido_delta["RMSE_Delta_baseline"]
        )
        barrido_delta.to_csv(
            RESULTS_DIR / "resumen_min_samples_leaf_fast_delta.csv",
            index=False,
        )

    # ========================================================
    # NUEVO: EVALUACION EXPLICITA DE FAST DELTA EN LOS 3 CONJUNTOS
    # Incluye baseline Delta=0 y errores extremos para cada configuracion.
    # Los conjuntos de prueba NO se usan para seleccionar el modelo.
    # ========================================================

    filas_fast_delta_evaluacion = []

    barrido_fast = metricas_df[
        (metricas_df["Objetivo"] == "Fast Delta (barrido)")
        & (metricas_df["Esquina"] == "Fast")
        & (metricas_df["Metrica_evaluada"] == "Delta")
        & metricas_df["Conjunto"].isin([
            "Validacion", "Test Labels", "Test Designs"
        ])
    ].copy()

    for conjunto_diag, datos_diag in [
        ("Validacion", train.iloc[indices_validacion]),
        ("Test Labels", test_labels),
        ("Test Designs", test_designs),
    ]:
        delta_real = datos_diag["Delta_Fast"].to_numpy(dtype=float)
        error_baseline = -delta_real  # baseline predice Delta = 0
        abs_baseline = np.abs(error_baseline)

        filas_fast_delta_evaluacion.append({
            "Modelo": "Baseline",
            "Experimento": "Delta=0",
            "Conjunto": conjunto_diag,
            "min_samples_leaf": np.nan,
            "MAE_Delta": float(np.mean(abs_baseline)),
            "RMSE_Delta": float(np.sqrt(np.mean(error_baseline ** 2))),
            "Error_absoluto_P95": float(np.percentile(abs_baseline, 95)),
            "Error_absoluto_P99": float(np.percentile(abs_baseline, 99)),
            "Error_absoluto_max": float(np.max(abs_baseline)),
            "N_error_abs_mayor_0_1": int(np.sum(abs_baseline > 0.1)),
            "N_error_abs_mayor_0_5": int(np.sum(abs_baseline > 0.5)),
            "N_error_abs_mayor_1": int(np.sum(abs_baseline > 1.0)),
        })

        for _, fila in barrido_fast[
            barrido_fast["Conjunto"] == conjunto_diag
        ].iterrows():
            parametros = json.loads(fila["Experimento"])
            hojas = int(parametros["min_samples_leaf"])
            modelo_nombre = fila["Modelo"]
            leaf_text = str(hojas).zfill(2)
            safe_model = modelo_nombre.lower().replace(" ", "_")
            detalle_path = RESULTS_DIR / (
                f"errores_fast_delta_{conjunto_diag.lower().replace(' ', '_')}_"
                f"{safe_model}_leaf_{leaf_text}.csv"
            )

            if not detalle_path.exists():
                continue

            detalle = pd.read_csv(detalle_path)
            abs_error = detalle["Error_absoluto"].to_numpy(dtype=float)

            filas_fast_delta_evaluacion.append({
                "Modelo": modelo_nombre,
                "Experimento": fila["Experimento"],
                "Conjunto": conjunto_diag,
                "min_samples_leaf": hojas,
                "MAE_Delta": float(np.mean(abs_error)),
                "RMSE_Delta": float(np.sqrt(np.mean(abs_error ** 2))),
                "Error_absoluto_P95": float(np.percentile(abs_error, 95)),
                "Error_absoluto_P99": float(np.percentile(abs_error, 99)),
                "Error_absoluto_max": float(np.max(abs_error)),
                "N_error_abs_mayor_0_1": int(np.sum(abs_error > 0.1)),
                "N_error_abs_mayor_0_5": int(np.sum(abs_error > 0.5)),
                "N_error_abs_mayor_1": int(np.sum(abs_error > 1.0)),
            })

    evaluacion_fast_delta_df = pd.DataFrame(filas_fast_delta_evaluacion)
    if not evaluacion_fast_delta_df.empty:
        baseline_rmse = (
            evaluacion_fast_delta_df[
                evaluacion_fast_delta_df["Modelo"] == "Baseline"
            ][["Conjunto", "RMSE_Delta"]]
            .rename(columns={"RMSE_Delta": "RMSE_Delta_baseline"})
        )
        evaluacion_fast_delta_df = evaluacion_fast_delta_df.merge(
            baseline_rmse,
            on="Conjunto",
            how="left",
        )
        evaluacion_fast_delta_df["Mejora_RMSE_vs_baseline_pct"] = np.where(
            evaluacion_fast_delta_df["RMSE_Delta_baseline"] > 0,
            (
                evaluacion_fast_delta_df["RMSE_Delta_baseline"]
                - evaluacion_fast_delta_df["RMSE_Delta"]
            ) / evaluacion_fast_delta_df["RMSE_Delta_baseline"] * 100,
            np.nan,
        )
        evaluacion_fast_delta_df["Mejora_RMSE_vs_baseline"] = (
            evaluacion_fast_delta_df["RMSE_Delta"]
            < evaluacion_fast_delta_df["RMSE_Delta_baseline"]
        )
        evaluacion_fast_delta_df.to_csv(
            RESULTS_DIR / "evaluacion_completa_fast_delta.csv",
            index=False,
        )

    # ========================================================
    # COMPARACION DEL DELTA CONTRA SU BASELINE
    # Se compara Delta real con Delta predicho.
    # El baseline predice siempre Delta = 0.
    # ========================================================

    filas_comparacion_delta = []

    for esquina in ("Slow", "Fast"):
        for conjunto in (
            "Train",
            "Validacion",
            "Test Labels",
            "Test Designs",
        ):
            # Baseline: predice Delta = 0
            filtro_baseline = (
                (metricas_df["Modelo"] == "Baseline")
                & (metricas_df["Objetivo"] == "Delta")
                & (metricas_df["Conjunto"] == conjunto)
                & (metricas_df["Esquina"] == esquina)
                & (metricas_df["Metrica_evaluada"] == "Delta")
            )

            # Modelo ganador: predice Delta
            filtro_modelo = (
                (metricas_df["Objetivo"] == "Delta (final)")
                & (metricas_df["Conjunto"] == conjunto)
                & (metricas_df["Esquina"] == esquina)
                & (metricas_df["Metrica_evaluada"] == "Delta")
            )

            baseline = metricas_df.loc[filtro_baseline]
            modelo = metricas_df.loc[filtro_modelo]

            if not baseline.empty and not modelo.empty:
                b = baseline.iloc[0]
                m = modelo.iloc[0]

                filas_comparacion_delta.append({
                    "Esquina": esquina,
                    "Conjunto": conjunto,
                    "Modelo": m["Modelo"],
                    "RMSE_Delta_baseline": b["RMSE"],
                    "RMSE_Delta_modelo": m["RMSE"],
                    "Mejora_porcentual": (
                        (b["RMSE"] - m["RMSE"]) / b["RMSE"] * 100
                        if b["RMSE"] != 0 else np.nan
                    ),
                    "Modelo_mejora_baseline": m["RMSE"] < b["RMSE"],
                })

    comparacion_delta_baseline = pd.DataFrame(
        filas_comparacion_delta
    )

    comparacion_delta_baseline.to_csv(
        RESULTS_DIR / "comparacion_delta_baseline.csv",
        index=False,
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

    
    # ============================================================
    # PARAMETROS DE CALIBRACION BASELINE-AWARE
    # ============================================================
    pd.DataFrame([
        {
            "Esquina": esquina,
            "Objetivo": objetivo,
            "Alpha": alpha,
            "Interpretacion": (
                "0 = baseline Typical; 1 = modelo ML completo; "
                "valores intermedios = mezcla"
            ),
        }
        for (esquina, objetivo), alpha in alphas_baseline_aware.items()
    ]).to_csv(
        RESULTS_DIR / "calibracion_baseline_aware.csv",
        index=False,
    )

    # ============================================================
    # PREDICCIONES DE TEST
    # ============================================================

    for conjunto, datos in conjuntos_test.items():

        pred_df = datos[
            COLUMNAS_LLAVE + [
                "Delay_Typical",
                "Delay_Slow",
                "Delay_Fast",
                "Delta_Slow",
                "Delta_Fast",
            ]
        ].copy()

        for esquina in ("Slow", "Fast"):

            # Baseline: usar directamente el delay Typical
            pred_df[f"Baseline_{esquina}"] = (
                datos["Delay_Typical"].to_numpy()
            )

            # Modelo entrenado para predecir el delay directamente
            pred_df[f"Predicted_{esquina}_Delay"] = (
                predicciones[(conjunto, esquina, "Delay")]["delay"]
            )

            # Modelo entrenado para predecir el delta
            pred_df[f"Predicted_{esquina}_Delta"] = (
                predicciones[(conjunto, esquina, "Delta")]["target"]
            )

            # Reconstrucción del delay a partir del delta predicho
            pred_df[f"Predicted_{esquina}_Delay_from_Delta"] = (
                predicciones[(conjunto, esquina, "Delta")]["delay"]
            )

            # Prediccion calibrada con alpha elegido en validacion.
            for objetivo_blend in ("Delay", "Delta"):
                alpha_blend = alphas_baseline_aware[(esquina, objetivo_blend)]
                pred_raw = predicciones[(conjunto, esquina, objetivo_blend)]["delay"]
                baseline_delay = datos["Delay_Typical"].to_numpy(dtype=float)
                delay_blend = (
                    baseline_delay
                    + alpha_blend
                    * (np.asarray(pred_raw, dtype=float) - baseline_delay)
                )

                if objetivo_blend == "Delay":
                    pred_df[
                        f"Predicted_{esquina}_Delay_BaselineAware"
                    ] = delay_blend
                else:
                    pred_df[
                        f"Predicted_{esquina}_Delta_BaselineAware"
                    ] = delay_blend - baseline_delay
                    pred_df[
                        f"Predicted_{esquina}_Delay_from_Delta_BaselineAware"
                    ] = delay_blend

                pred_df[
                    f"Alpha_{esquina}_{objetivo_blend}_BaselineAware"
                ] = alpha_blend

        nombre_archivo = (
            f"predicciones_{conjunto.lower().replace(' ', '_')}.csv"
        )

        pred_df.to_csv(
            RESULTS_DIR / nombre_archivo,
            index=False,
        )

        print(f"Predicciones guardadas: {RESULTS_DIR / nombre_archivo}")




    # ========================================================
    # TABLA DE 4 CONJUNTOS
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
                    metricas_df["Objetivo"].isin(
                        [f"{objetivo} (final)", objetivo]
                    )
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

            objetivos_busqueda = (
                [f"{objetivo} (final)", objetivo]
                if objetivo_final
                else [objetivo]
            )

            filtro_base = (
                (metricas_df["Modelo"] == modelo)
                & (metricas_df["Experimento"] == experimento)
                & (metricas_df["Conjunto"] == conjunto)
                & (metricas_df["Esquina"] == esquina)
                & (metricas_df["Metrica_evaluada"] == "Delay reconstruido")
            )

            encontrados = metricas_df.loc[
                filtro_base
                & metricas_df["Objetivo"].isin(objetivos_busqueda)
            ]

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
        " - comparacion_min_samples_leaf_fast_delta.csv"
    )

    print(
        " - resumen_min_samples_leaf_fast_delta.csv"
    )

    print(
        " - evaluacion_completa_fast_delta.csv"
    )

    print(
        " - comparacion_hybrid_rf_baseline_fast_delta.csv"
    )

    print(
        " - modelo_hibrido_fast_delta.joblib"
    )

    print(
        " - errores_fast_delta_*.csv"
    )

    print(
        " - errores_por_diseno_fast_delta_*.csv"
    )

    print(
        " - hist_error_fast_delta_*.png"
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
    # Experimento adicional solicitado: clasificador de Delta casi cero + regresor.
    ejecutar_experimento_clasificador_regresor(
        train=train,
        conjuntos_test=conjuntos_test,
        columnas_features=columnas_features,
        indices_train=indices_train,
        indices_validacion=indices_validacion,
        tolerancia=0.05,
        metricas_df=metricas_df,
    )

    print("PROCESO TERMINADO")
    print("=" * 70)


# ============================================================
# EJECUCION
# ============================================================

if __name__ == "__main__":
    main()
