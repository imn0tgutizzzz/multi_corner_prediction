import pandas as pd
from pathlib import Path


# Carpeta donde están los CSV

BASE_DIR = Path(__file__).resolve().parent


# Recuperar Label Delay

def obtener_label_delay(df, nombre):
    """
    data.py eliminó Label Delay después de calcular:

        Delta = Label Delay - Delay

    Entonces se puede recuperar mediante:

        Label Delay = Delta + Delay
    """

    columnas = ["Delay", "Delta"]

    for col in columnas:
        if col not in df.columns:
            raise ValueError(
                f'La columna "{col}" no existe en {nombre}'
            )

    return df["Delay"] + df["Delta"]


# Crear dataset multicorner

def crear_multicorner(
    typical_file,
    slow_file,
    fast_file,
    output_file
):

    print("\n")
    print("=" * 60)
    print(f"CREANDO: {output_file}")
    print("=" * 60)

    # Cargar datos

    typical = pd.read_csv(BASE_DIR / typical_file)
    slow = pd.read_csv(BASE_DIR / slow_file)
    fast = pd.read_csv(BASE_DIR / fast_file)

    print(f"Typical: {len(typical)} filas")
    print(f"Slow:    {len(slow)} filas")
    print(f"Fast:    {len(fast)} filas")

    # Claves para hacer el matching

    keys = [
        "Description",
        "Previous_description"
    ]

    # Si existe Design, también se utiliza.
    if "Design" in typical.columns:
        keys = [
            "Design",
            "Description",
            "Previous_description"
        ]

    print("\nClaves utilizadas para matching:")
    print(keys)

    # Verificar que las columnas existan

    for nombre, df in [
        ("Typical", typical),
        ("Slow", slow),
        ("Fast", fast)
    ]:
        for key in keys:
            if key not in df.columns:
                raise ValueError(
                    f'La columna "{key}" no existe en {nombre}'
                )

    # Comprobar duplicados

    print("\nComprobando claves duplicadas...")

    for nombre, df in [
        ("Typical", typical),
        ("Slow", slow),
        ("Fast", fast)
    ]:

        duplicados = df.duplicated(
            subset=keys
        ).sum()

        print(
            f"{nombre}: {duplicados} claves duplicadas"
        )

        if duplicados > 0:
            raise ValueError(
                f"Hay correspondencias ambiguas en {nombre}"
            )

    # Recuperar Label Delay

    typical["Label_Delay_Typical"] = (
        obtener_label_delay(
            typical,
            "Typical"
        )
    )

    slow["Label_Delay_Slow"] = (
        obtener_label_delay(
            slow,
            "Slow"
        )
    )

    fast["Label_Delay_Fast"] = (
        obtener_label_delay(
            fast,
            "Fast"
        )
    )

    # Verificar correspondencia entre corners

    typical_keys = set(
        map(tuple, typical[keys].values)
    )

    slow_keys = set(
        map(tuple, slow[keys].values)
    )

    fast_keys = set(
        map(tuple, fast[keys].values)
    )

    print("\nCorrespondencia:")

    print(
        "Typical = Slow:",
        typical_keys == slow_keys
    )

    print(
        "Typical = Fast:",
        typical_keys == fast_keys
    )

    print(
        "Los tres coinciden:",
        typical_keys == slow_keys == fast_keys
    )

    # Encontrar solamente muestras presentes en los 3 corners

    comunes = (
        typical_keys
        & slow_keys
        & fast_keys
    )

    print("\nMuestras completas:")
    print(len(comunes))

    print("\nMuestras sin correspondencia completa:")

    print(
        "Typical:",
        len(typical_keys - comunes)
    )

    print(
        "Slow:",
        len(slow_keys - comunes)
    )

    print(
        "Fast:",
        len(fast_keys - comunes)
    )

    # Seleccionar labels Slow y Fast

    slow_labels = slow[
        keys + ["Label_Delay_Slow"]
    ].copy()

    fast_labels = fast[
        keys + ["Label_Delay_Fast"]
    ].copy()

    # Matching

    result = typical.merge(
        slow_labels,
        on=keys,
        how="inner",
        validate="one_to_one"
    )

    result = result.merge(
        fast_labels,
        on=keys,
        how="inner",
        validate="one_to_one"
    )

    # Poner los tres labels al final

    label_columns = [
        "Label_Delay_Typical",
        "Label_Delay_Slow",
        "Label_Delay_Fast"
    ]

    feature_columns = [
        col
        for col in result.columns
        if col not in label_columns
    ]

    result = result[
        feature_columns + label_columns
    ]

    # Validaciones finales

    print("\nRESULTADO:")

    print(
        "Filas Typical:",
        len(typical)
    )

    print(
        "Filas finales:",
        len(result)
    )

    print(
        "Filas eliminadas:",
        len(typical) - len(result)
    )

    print("\nValores faltantes:")

    print(
        result[
            label_columns
        ].isna().sum()
    )

    # Guardar archivo

    output_path = BASE_DIR / output_file

    result.to_csv(
        output_path,
        index=False
    )

    print("\nArchivo generado:")
    print(output_path)


# MAIN

if __name__ == "__main__":

    # TRAIN

    crear_multicorner(
        "treated_labels_train_typical.csv",
        "treated_labels_train_slow.csv",
        "treated_labels_train_fast.csv",
        "train_multicorner.csv"
    )

    # TEST LABELS

    crear_multicorner(
        "treated_labels_typical.csv",
        "treated_labels_slow.csv",
        "treated_labels_fast.csv",
        "test_labels_multicorner.csv"
    )

    # TEST DESIGNS

    crear_multicorner(
        "treated_test_designs_typical.csv",
        "treated_test_designs_slow.csv",
        "treated_test_designs_fast.csv",
        "test_designs_multicorner.csv"
    )