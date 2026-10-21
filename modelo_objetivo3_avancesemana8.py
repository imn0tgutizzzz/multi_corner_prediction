from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
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
    raise SystemExit("Instala XGBoost con: python -m pip install xgboost") from exc

# ============================================================
# OBJETIVO 3 - ML MULTICORNER
#
# Se comparan:
#   1) Delay completo: modelo -> Delay Slow/Fast
#   2) Delta: modelo -> Delta Slow/Fast
#      y luego Delay_pred = Delay_Typical + Delta_pred
#
# Delay_Typical se usa como feature porque se supone conocido.
# Delta original NO se usa como feature para evitar leakage.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "resultados_objetivo3"
GRAFICAS_DIR = RESULTS_DIR / "graficas"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SVR_MAX_TRAIN = 12000
PLOT_MAX_POINTS = 15000

COLUMNAS_LLAVE = ["Previous_description", "Description"]
COLUMNAS_EXCLUIR = {
    "row_id", "Design", "Path #", "Previous_description", "Description",
    "Delta", "Delay_Slow", "Delay_Fast", "Delta_Slow", "Delta_Fast",
}


def cargar_multicorner(prefijo):
    archivos = {}
    for esquina in ("typical", "slow", "fast"):
        ruta = BASE_DIR / f"{prefijo}_{esquina}.csv"
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontro el archivo:\n{ruta}")
        df = pd.read_csv(ruta)
        df.columns = df.columns.str.strip()
        archivos[esquina] = df
        print(f"{ruta.name}: {df.shape}")

    llaves = COLUMNAS_LLAVE.copy()
    if all("Design" in df.columns for df in archivos.values()):
        llaves = ["Design"] + llaves

    for nombre, df in archivos.items():
        faltantes = set(llaves + ["Delay"]) - set(df.columns)
        if faltantes:
            raise ValueError(f"{nombre}: faltan columnas {faltantes}")
        if df[llaves].isna().any().any():
            raise ValueError(f"{nombre}: hay llaves nulas")
        if df.duplicated(llaves).any():
            raise ValueError(f"{nombre}: hay llaves duplicadas")

    datos = archivos["typical"].rename(columns={"Delay": "Delay_Typical"}).copy()

    for nombre in ("slow", "fast"):
        etiqueta = archivos[nombre][llaves + ["Delay"]].rename(columns={"Delay": f"Delay_{nombre.title()}"})
        datos = datos.merge(etiqueta, on=llaves, how="left", validate="one_to_one", sort=False, indicator=True)
        faltantes = (datos["_merge"] != "both").sum()
        datos = datos.drop(columns="_merge")
        if faltantes:
            raise ValueError(f"{prefijo}: {faltantes} segmentos sin correspondencia en {nombre}")

    for esquina in ("Typical", "Slow", "Fast"):
        col = f"Delay_{esquina}"
        datos[col] = pd.to_numeric(datos[col], errors="coerce")
        if not np.isfinite(datos[col]).all():
            raise ValueError(f"{prefijo}: {col} tiene NaN o infinito")

    datos["Delta_Slow"] = datos["Delay_Slow"] - datos["Delay_Typical"]
    datos["Delta_Fast"] = datos["Delay_Fast"] - datos["Delay_Typical"]
    print(f"Segmentos alineados: {len(datos):,}")
    return datos


def metricas(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.isfinite(y_real).all() or not np.isfinite(y_pred).all():
        raise ValueError("Hay valores no finitos en reales o predicciones")
    pearson = np.nan
    if len(y_real) > 1 and np.std(y_real) > 0 and np.std(y_pred) > 0:
        pearson = np.corrcoef(y_real, y_pred)[0, 1]
    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "R2": r2_score(y_real, y_pred),
        "Pearson": pearson,
    }


def registrar(resultados, modelo, experimento, objetivo, conjunto, esquina,
               y_real, y_pred, n_entrenamiento=None, metrica_evaluada="Delay"):
    resultados.append({
        "Modelo": modelo,
        "Experimento": experimento,
        "Objetivo": objetivo,
        "Conjunto": conjunto,
        "Esquina": esquina,
        "Metrica_evaluada": metrica_evaluada,
        "N_entrenamiento": n_entrenamiento,
        "N_evaluacion": len(y_real),
        **metricas(y_real, y_pred),
    })


def crear_modelo(nombre, parametros):
    if nombre == "Ridge":
        pasos = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("modelo", Ridge(**parametros)),
        ]
    elif nombre == "SVR":
        pasos = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("modelo", SVR(**parametros)),
        ]
    elif nombre == "Random Forest":
        pasos = [
            ("imputer", SimpleImputer(strategy="median")),
            ("modelo", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **parametros)),
        ]
    elif nombre == "Gradient Boosting":
        pasos = [
            ("imputer", SimpleImputer(strategy="median")),
            ("modelo", GradientBoostingRegressor(random_state=RANDOM_STATE, **parametros)),
        ]
    elif nombre == "Hist Gradient Boosting":
        pasos = [
            ("imputer", SimpleImputer(strategy="median")),
            ("modelo", HistGradientBoostingRegressor(random_state=RANDOM_STATE, **parametros)),
        ]
    elif nombre == "XGBoost":
        pasos = [
            ("imputer", SimpleImputer(strategy="median")),
            ("modelo", XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                                    n_jobs=-1, tree_method="hist", **parametros)),
        ]
    else:
        raise ValueError(f"Modelo desconocido: {nombre}")
    return Pipeline(pasos)


def crear_experimentos():
    experimentos = [
        ("Ridge", {"alpha": 1.0}),
        ("Gradient Boosting", {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3}),
        ("Hist Gradient Boosting", {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31}),
        ("SVR", {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}),
    ]
    for n in (10, 50, 100, 200, 300, 500):
        experimentos.append(("Random Forest", {"n_estimators": n, "max_depth": 20, "min_samples_leaf": 2}))
    for d in (10, 30, None):
        experimentos.append(("Random Forest", {"n_estimators": 100, "max_depth": d, "min_samples_leaf": 2}))
    for leaf in (1, 4):
        experimentos.append(("Random Forest", {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": leaf}))
    for n in (100, 200, 300, 500):
        experimentos.append(("XGBoost", {
            "n_estimators": n, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.9, "colsample_bytree": 0.9,
        }))
    return experimentos


def preparar_X(datos, columnas_features):
    faltantes = set(columnas_features) - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan variables: {faltantes}")
    X = datos[columnas_features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().all().any():
        raise ValueError(f"Variables completamente vacias: {X.columns[X.isna().all()].tolist()}")
    return X


def predecir_target(modelo, X):
    return modelo.predict(X)


def predecir_delay(modelo, X, objetivo):
    pred = modelo.predict(X)
    if objetivo == "Delta":
        pred = X["Delay_Typical"].to_numpy() + pred
    return pred


def guardar_grafica(y_real, y_pred, titulo, ruta):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y_real), size=min(len(y_real), PLOT_MAX_POINTS), replace=False)
    lo = min(y_real.min(), y_pred.min())
    hi = max(y_real.max(), y_pred.max())
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_real[idx], y_pred[idx], s=8, alpha=0.3)
    ax.plot([lo, hi], [lo, hi], "--", color="black")
    ax.set_xlabel("Delay real")
    ax.set_ylabel("Delay predicho")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=180)
    plt.close(fig)


def guardar_importancias(datos, ruta, titulo):
    datos = datos.sort_values("Importancia", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(datos["Feature"], datos["Importancia"])
    ax.set_xlabel("Importancia relativa")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=180)
    plt.close(fig)


def diagnostico_deltas(train, test_labels, test_designs):
    filas = []
    for nombre, datos in (("Train", train), ("Test Labels", test_labels), ("Test Designs", test_designs)):
        for esquina in ("Slow", "Fast"):
            d = datos[f"Delta_{esquina}"].to_numpy(float)
            filas.append({
                "Conjunto": nombre, "Esquina": esquina, "N": len(d),
                "Media_Delta": np.mean(d), "Desv_Delta": np.std(d),
                "Min_Delta": np.min(d), "Mediana_Delta": np.median(d), "Max_Delta": np.max(d),
                "RMSE_Delta_baseline": np.sqrt(np.mean(d ** 2)),
            })
    df = pd.DataFrame(filas)
    df.to_csv(RESULTS_DIR / "diagnostico_distribucion_delta.csv", index=False)
    return df


def main():
    print("=" * 70)
    print("OBJETIVO 3 - MACHINE LEARNING MULTICORNER")
    print("=" * 70)

    print("\nCARGANDO TRAIN")
    train = cargar_multicorner("treated_labels_train")
    print("\nCARGANDO TEST LABELS")
    test_labels = cargar_multicorner("treated_labels")
    print("\nCARGANDO TEST DESIGNS")
    test_designs = cargar_multicorner("treated_test_designs")

    conjuntos_test = {"Test Labels": test_labels, "Test Designs": test_designs}

    print("\nDIAGNOSTICO DE DELTAS")
    diag = diagnostico_deltas(train, test_labels, test_designs)
    print(diag.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    columnas_features = [c for c in train.columns if c not in COLUMNAS_EXCLUIR]
    if "Delay_Typical" not in columnas_features:
        raise ValueError("Delay_Typical debe ser una variable de entrada")

    print("\nVARIABLES UTILIZADAS")
    for c in columnas_features:
        print(" -", c)
    print("Total:", len(columnas_features))
    pd.DataFrame({"Feature": columnas_features}).to_csv(RESULTS_DIR / "features_utilizadas.csv", index=False)

    X = preparar_X(train, columnas_features)
    X_tests = {nombre: preparar_X(datos, columnas_features) for nombre, datos in conjuntos_test.items()}

    indices_train, indices_validacion = train_test_split(
        np.arange(len(train)), test_size=0.20, random_state=RANDOM_STATE
    )
    X_train = X.iloc[indices_train]
    X_validacion = X.iloc[indices_validacion]
    print("\nDIVISION DE DATOS")
    print("Train:", len(X_train))
    print("Validacion:", len(X_validacion))

    resultados = []
    mejores_modelos = []
    importancias = []
    predicciones = {}

    # ========================================================
    # BASELINE
    # ========================================================
    print("\nCALCULANDO BASELINE")
    print("Baseline: Delay predicho = Delay_Typical")
    conjuntos_base = [
        ("Train", train.iloc[indices_train]),
        ("Validacion", train.iloc[indices_validacion]),
        ("Test Labels", test_labels),
        ("Test Designs", test_designs),
    ]
    for conjunto, datos in conjuntos_base:
        for esquina in ("Slow", "Fast"):
            registrar(resultados, "Baseline", "Typical como prediccion", "Baseline", conjunto, esquina,
                      datos[f"Delay_{esquina}"], datos["Delay_Typical"], metrica_evaluada="Delay")
            registrar(resultados, "Baseline", "Delta=0", "Baseline Delta", conjunto, esquina,
                      datos[f"Delta_{esquina}"], np.zeros(len(datos)), metrica_evaluada="Delta")

    experimentos = crear_experimentos()

    # ========================================================
    # MODELOS
    # ========================================================
    for esquina in ("Slow", "Fast"):
        for objetivo in ("Delay", "Delta"):
            print("\n" + "=" * 70)
            print(f"ESQUINA: {esquina}")
            print(f"OBJETIVO: {objetivo}")
            print("=" * 70)

            columna_objetivo = f"{objetivo}_{esquina}"
            y_train = train[columna_objetivo].iloc[indices_train]
            y_validacion = train[columna_objetivo].iloc[indices_validacion]
            real_delay_validacion = train[f"Delay_{esquina}"].iloc[indices_validacion]
            candidatos = []

            for nombre_modelo, parametros in experimentos:
                experimento = json.dumps(parametros, sort_keys=True)
                print(f"\nEntrenando {nombre_modelo} | {esquina} | {objetivo} | {experimento}", flush=True)
                modelo = crear_modelo(nombre_modelo, parametros)

                if nombre_modelo == "SVR" and len(X_train) > SVR_MAX_TRAIN:
                    rng = np.random.default_rng(RANDOM_STATE)
                    idx_sub = np.sort(rng.choice(len(X_train), size=SVR_MAX_TRAIN, replace=False))
                    X_ajuste = X_train.iloc[idx_sub]
                    y_ajuste = y_train.iloc[idx_sub]
                else:
                    X_ajuste = X_train
                    y_ajuste = y_train

                modelo.fit(X_ajuste, y_ajuste)

                pred_target_train = predecir_target(modelo, X_train)
                pred_target_validacion = predecir_target(modelo, X_validacion)
                pred_delay_train = predecir_delay(modelo, X_train, objetivo)
                pred_delay_validacion = predecir_delay(modelo, X_validacion, objetivo)

                # Guardamos tanto la metrica del target puro como la del Delay final.
                registrar(resultados, nombre_modelo, experimento, objetivo, "Train", esquina,
                          y_train, pred_target_train, len(X_ajuste), objetivo)
                registrar(resultados, nombre_modelo, experimento, objetivo, "Validacion", esquina,
                          y_validacion, pred_target_validacion, len(X_ajuste), objetivo)
                registrar(resultados, nombre_modelo, experimento, objetivo, "Train", esquina,
                          train[f"Delay_{esquina}"].iloc[indices_train], pred_delay_train,
                          len(X_ajuste), "Delay reconstruido")
                registrar(resultados, nombre_modelo, experimento, objetivo, "Validacion", esquina,
                          real_delay_validacion, pred_delay_validacion, len(X_ajuste), "Delay reconstruido")

                # Para escoger el mejor modelo usamos el RMSE del Delay FINAL.
                # Esto hace comparable Delay directo contra Delta reconstruido.
                rmse_delay = metricas(real_delay_validacion, pred_delay_validacion)["RMSE"]
                candidatos.append((rmse_delay, nombre_modelo, parametros, experimento))

            mejor_rmse, mejor_nombre, mejores_parametros, mejor_experimento = min(candidatos, key=lambda x: x[0])

            print("\nMEJOR MODELO")
            print("Esquina:", esquina)
            print("Objetivo:", objetivo)
            print("Modelo:", mejor_nombre)
            print(f"RMSE Delay validacion: {mejor_rmse:.6f}")

            mejores_modelos.append({
                "Esquina": esquina,
                "Objetivo": objetivo,
                "Modelo": mejor_nombre,
                "Experimento": mejor_experimento,
                "RMSE_Delay_validacion": mejor_rmse,
            })

            # Entrenamiento final con todo train.
            modelo_final = crear_modelo(mejor_nombre, mejores_parametros)
            y_total = train[columna_objetivo]
            if mejor_nombre == "SVR" and len(X) > SVR_MAX_TRAIN:
                rng = np.random.default_rng(RANDOM_STATE)
                idx_sub = np.sort(rng.choice(len(X), size=SVR_MAX_TRAIN, replace=False))
                X_final = X.iloc[idx_sub]
                y_final = y_total.iloc[idx_sub]
            else:
                X_final = X
                y_final = y_total
            modelo_final.fit(X_final, y_final)

            nombre_archivo = f"{esquina.lower()}_{objetivo.lower()}"
            joblib.dump(modelo_final, RESULTS_DIR / f"mejor_modelo_{nombre_archivo}.joblib")

            pred_train_target = predecir_target(modelo_final, X)
            pred_train_delay = predecir_delay(modelo_final, X, objetivo)
            registrar(resultados, mejor_nombre, mejor_experimento, f"{objetivo} (final)", "Train final", esquina,
                      train[columna_objetivo], pred_train_target, len(X_final), objetivo)
            registrar(resultados, mejor_nombre, mejor_experimento, f"{objetivo} (final)", "Train final", esquina,
                      train[f"Delay_{esquina}"], pred_train_delay, len(X_final), "Delay reconstruido")

            for conjunto, datos in conjuntos_test.items():
                pred_target = predecir_target(modelo_final, X_tests[conjunto])
                pred_delay = predecir_delay(modelo_final, X_tests[conjunto], objetivo)

                registrar(resultados, mejor_nombre, mejor_experimento, f"{objetivo} (final)", conjunto, esquina,
                          datos[columna_objetivo], pred_target, len(X_final), objetivo)
                registrar(resultados, mejor_nombre, mejor_experimento, f"{objetivo} (final)", conjunto, esquina,
                          datos[f"Delay_{esquina}"], pred_delay, len(X_final), "Delay reconstruido")

                predicciones[(conjunto, esquina, objetivo)] = pred_delay

                guardar_grafica(
                    datos[f"Delay_{esquina}"], pred_delay,
                    f"{conjunto}: {esquina} {objetivo} - {mejor_nombre}",
                    GRAFICAS_DIR / f"{conjunto.lower().replace(' ', '_')}_{nombre_archivo}.png",
                )

            # Importancia del ganador si es un modelo de arboles.
            estimador = modelo_final.named_steps["modelo"]
            if hasattr(estimador, "feature_importances_"):
                for variable, importancia in zip(columnas_features, estimador.feature_importances_):
                    importancias.append({
                        "Esquina": esquina, "Objetivo": objetivo,
                        "Modelo": mejor_nombre, "Feature": variable,
                        "Importancia": importancia,
                    })

            # Importancia RF de referencia, como pidio el profesor.
            rf_ref = crear_modelo("Random Forest", {
                "n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2,
            })
            rf_ref.fit(X_train, y_train)
            valores = rf_ref.named_steps["modelo"].feature_importances_
            for variable, importancia in zip(columnas_features, valores):
                importancias.append({
                    "Esquina": esquina, "Objetivo": objetivo,
                    "Modelo": "Random Forest referencia", "Feature": variable,
                    "Importancia": importancia,
                })

    # ========================================================
    # GUARDAR RESULTADOS
    # ========================================================
    metricas_df = pd.DataFrame(resultados)
    metricas_df.to_csv(RESULTS_DIR / "metricas_objetivo3.csv", index=False)

    mejores_df = pd.DataFrame(mejores_modelos)
    mejores_df.to_csv(RESULTS_DIR / "mejores_modelos.csv", index=False)

    metricas_df[
        (metricas_df["Conjunto"] == "Validacion") &
        (metricas_df["Modelo"] != "Baseline")
    ].to_csv(RESULTS_DIR / "experimentos_validacion.csv", index=False)

    importancias_df = pd.DataFrame(importancias)
    importancias_df.to_csv(RESULTS_DIR / "importancia_variables.csv", index=False)

    for (esquina, objetivo, modelo), grupo in importancias_df.groupby(["Esquina", "Objetivo", "Modelo"]):
        nombre = f"importancia_{esquina.lower()}_{objetivo.lower()}_{modelo.lower().replace(' ', '_')}.png"
        guardar_importancias(grupo, GRAFICAS_DIR / nombre, f"Importancia: {esquina}, {objetivo}, {modelo}")

    # ========================================================
    # PREDICCIONES DE TEST
    # ========================================================
    for conjunto, datos in conjuntos_test.items():
        columnas = ["Delay_Typical", "Delay_Slow", "Delay_Fast"]
        for columna in ("row_id", "Design", "Previous_description", "Description"):
            if columna in datos.columns:
                columnas.insert(0, columna)
        pred_df = datos[columnas].copy()

        for esquina in ("Slow", "Fast"):
            pred_df[f"Baseline_{esquina}"] = datos["Delay_Typical"].to_numpy()
            for objetivo in ("Delay", "Delta"):
                pred_df[f"Predicted_{esquina}_{objetivo}"] = predicciones[(conjunto, esquina, objetivo)]
            guardar_grafica(
                datos[f"Delay_{esquina}"], datos["Delay_Typical"],
                f"{conjunto}: Baseline {esquina}",
                GRAFICAS_DIR / f"{conjunto.lower().replace(' ', '_')}_baseline_{esquina.lower()}.png",
            )

        pred_df.to_csv(
            RESULTS_DIR / f"predicciones_{conjunto.lower().replace(' ', '_')}.csv",
            index=False,
        )

    # ========================================================
    # RESUMEN PARA LA REUNION
    # ========================================================
    filas = []
    for _, mejor in mejores_df.iterrows():
        esquina = mejor["Esquina"]
        objetivo = mejor["Objetivo"]
        modelo = mejor["Modelo"]
        experimento = mejor["Experimento"]

        def obtener_rmse(conjunto):
            f = (
                (metricas_df["Modelo"] == modelo) &
                (metricas_df["Experimento"] == experimento) &
                (metricas_df["Objetivo"] == f"{objetivo} (final)") &
                (metricas_df["Conjunto"] == conjunto) &
                (metricas_df["Esquina"] == esquina) &
                (metricas_df["Metrica_evaluada"] == "Delay reconstruido")
            )
            valores = metricas_df.loc[f, "RMSE"]
            return valores.iloc[0] if len(valores) else np.nan

        filas.append({
            "Esquina": esquina,
            "Objetivo": objetivo,
            "Mejor_modelo": modelo,
            "Experimento": experimento,
            "RMSE_validacion_Delay": mejor["RMSE_Delay_validacion"],
            "RMSE_test_labels_Delay": obtener_rmse("Test Labels"),
            "RMSE_test_designs_Delay": obtener_rmse("Test Designs"),
        })

    resumen = pd.DataFrame(filas)
    resumen.to_csv(RESULTS_DIR / "resumen_para_reunion.csv", index=False)

    print("\n" + "=" * 70)
    print("MEJORES MODELOS")
    print("=" * 70)
    print(mejores_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n" + "=" * 70)
    print("RESUMEN PARA LA REUNION")
    print("=" * 70)
    print(resumen.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nArchivos guardados en:")
    print(RESULTS_DIR)
    print("\nPROCESO TERMINADO")


if __name__ == "__main__":
    main()
