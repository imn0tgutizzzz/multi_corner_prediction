import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_FILE = (
    BASE_DIR
    / "resultados_objetivo3"
    / "predicciones_test_designs.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "resultados_objetivo3"
    / "graficas"
)

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# FUNCIÓN PARA CREAR GRÁFICA
# ============================================================

def crear_scatter(
    df,
    x_col,
    y_col,
    titulo,
    xlabel,
    ylabel,
    filename
):

    datos = df[[x_col, y_col]].dropna()

    x = datos[x_col]
    y = datos[y_col]

    correlacion = x.corr(y)
    rmse = mean_squared_error(y, x) ** 0.5
    r2 = r2_score(y, x)

    minimo = min(x.min(), y.min())
    maximo = max(x.max(), y.max())

    plt.figure(figsize=(8, 6))

    plt.scatter(
        x,
        y,
        s=8,
        alpha=0.30
    )

    plt.plot(
        [minimo, maximo],
        [minimo, maximo],
        linestyle="--",
        linewidth=2,
        label="Referencia y = x"
    )

    plt.xlim(minimo, maximo)
    plt.ylim(minimo, maximo)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(
        f"{titulo}\n"
        f"RMSE = {rmse:.4f} | "
        f"Pearson = {correlacion:.4f} | "
        f"R² = {r2:.4f}"
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Generada: {output_path}")


# ============================================================
# CARGAR DATOS
# ============================================================

df = pd.read_csv(PREDICTIONS_FILE)

print("Filas:", len(df))


# ============================================================
# BASELINE SLOW: TYPICAL VS SLOW REAL
# ============================================================

crear_scatter(
    df=df,
    x_col="Baseline_Slow",
    y_col="Delay_Slow",
    titulo="Test Designs: Baseline Typical vs Slow real",
    xlabel="Baseline Slow",
    ylabel="Slow real",
    filename="test_designs_baseline_slow_vs_real.png"
)


# ============================================================
# BASELINE FAST: TYPICAL VS FAST REAL
# ============================================================

crear_scatter(
    df=df,
    x_col="Baseline_Fast",
    y_col="Delay_Fast",
    titulo="Test Designs: Baseline Typical vs Fast real",
    xlabel="Baseline Fast",
    ylabel="Fast real",
    filename="test_designs_baseline_fast_vs_real.png"
)


# ============================================================
# SLOW PREDICHO VS SLOW REAL
# ============================================================

crear_scatter(
    df=df,
    x_col="RF_Slow",
    y_col="Delay_Slow",
    titulo="Test Designs: Slow predicho vs Slow real",
    xlabel="Slow predicho",
    ylabel="Slow real",
    filename="test_designs_slow_predicho_vs_real.png"
)


# ============================================================
# FAST PREDICHO VS FAST REAL
# ============================================================

crear_scatter(
    df=df,
    x_col="RF_Fast",
    y_col="Delay_Fast",
    titulo="Test Designs: Fast predicho vs Fast real",
    xlabel="Fast predicho",
    ylabel="Fast real",
    filename="test_designs_fast_predicho_vs_real.png"
)


print("\nProceso terminado.")
print(f"Gráficas guardadas en:\n{OUTPUT_DIR}")
