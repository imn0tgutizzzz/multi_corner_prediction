import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "graficas_objetivo_2"
OUTPUT_DIR.mkdir(exist_ok=True)

datasets = {
    "Train": BASE_DIR / "train_multicorner.csv",
    "Test Labels": BASE_DIR / "test_labels_multicorner.csv",
    "Test Designs": BASE_DIR / "test_designs_multicorner.csv",
}


# Columnas generadas en los CSV multicorner
TYPICAL = "Label_Delay_Typical"
SLOW = "Label_Delay_Slow"
FAST = "Label_Delay_Fast"


# ============================================================
# FUNCIÓN PARA CREAR UNA GRÁFICA
# ============================================================

def crear_scatter(
    df,
    x_col,
    y_col,
    dataset_name,
    corner_name,
    filename
):

    # Eliminar únicamente filas inválidas para estas columnas
    plot_df = df[[x_col, y_col]].dropna()

    x = plot_df[x_col]
    y = plot_df[y_col]

    # --------------------------------------------------------
    # Correlación de Pearson
    # --------------------------------------------------------

    correlacion = x.corr(y)

    # --------------------------------------------------------
    # Límites comunes para ambos ejes
    # --------------------------------------------------------
    #
    # Se utilizan los mismos límites en X y Y para que la
    # recta y = x represente correctamente los puntos donde
    # Typical y el corner analizado tienen exactamente el
    # mismo valor.
    # --------------------------------------------------------

    minimo = min(x.min(), y.min())
    maximo = max(x.max(), y.max())

    # --------------------------------------------------------
    # Crear figura
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        x,
        y,
        s=8,
        alpha=0.30
    )

    # --------------------------------------------------------
    # Línea de referencia y = x
    # --------------------------------------------------------
    #
    # Esta línea representa igualdad perfecta entre Typical
    # y el corner analizado.
    #
    # Por ejemplo:
    #
    # Typical = Slow  -> punto sobre y = x
    # Typical = Fast  -> punto sobre y = x
    #
    # Ya no se calcula un ajuste lineal sobre los datos.
    # --------------------------------------------------------

    plt.plot(
        [minimo, maximo],
        [minimo, maximo],
        linestyle="--",
        linewidth=2,
        label="Referencia y = x"
    )

    # Mismos límites para ambos ejes
    plt.xlim(minimo, maximo)
    plt.ylim(minimo, maximo)

    plt.xlabel("Label Delay - Typical")
    plt.ylabel(f"Label Delay - {corner_name}")

    plt.title(
        f"{dataset_name}: Typical vs {corner_name}\n"
        f"Correlación de Pearson = {correlacion:.4f}"
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # --------------------------------------------------------
    # Guardar figura
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Generada: {output_path}")

    return correlacion


# ============================================================
# PROCESAR LOS TRES DATASETS
# ============================================================

print("ANÁLISIS DE RELACIÓN ENTRE CORNERS")

resultados = []

for dataset_name, file_path in datasets.items():

    print("\n" + "=" * 50)
    print(dataset_name)
    print("=" * 50)

    df = pd.read_csv(file_path)

    print("Filas:", len(df))

    # --------------------------------------------------------
    # Verificación de columnas
    # --------------------------------------------------------

    required_columns = [
        TYPICAL,
        SLOW,
        FAST
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f'No existe la columna "{column}" '
                f'en {file_path.name}'
            )

    # Nombre utilizado para guardar las imágenes
    nombre_archivo = (
        dataset_name
        .lower()
        .replace(" ", "_")
    )

    # --------------------------------------------------------
    # Typical vs Slow
    # --------------------------------------------------------

    corr_slow = crear_scatter(
        df=df,
        x_col=TYPICAL,
        y_col=SLOW,
        dataset_name=dataset_name,
        corner_name="Slow",
        filename=(
            f"{nombre_archivo}_typical_vs_slow.png"
        )
    )

    # --------------------------------------------------------
    # Typical vs Fast
    # --------------------------------------------------------

    corr_fast = crear_scatter(
        df=df,
        x_col=TYPICAL,
        y_col=FAST,
        dataset_name=dataset_name,
        corner_name="Fast",
        filename=(
            f"{nombre_archivo}_typical_vs_fast.png"
        )
    )

    resultados.append({
        "Dataset": dataset_name,
        "Typical_vs_Slow": corr_slow,
        "Typical_vs_Fast": corr_fast
    })


# ============================================================
# RESUMEN NUMÉRICO
# ============================================================

resultados_df = pd.DataFrame(resultados)

print("\n" + "=" * 50)
print("CORRELACIONES")
print("=" * 50)

print(
    resultados_df.to_string(
        index=False
    )
)

resultados_df.to_csv(
    OUTPUT_DIR / "correlaciones.csv",
    index=False
)

print("\nProceso terminado.")
print(
    f"Resultados guardados en:\n{OUTPUT_DIR}"
)