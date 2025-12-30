import pandas as pd

SITES = ['AR', 'PI', 'BA']

for site in SITES:
    print(f"Procesando sitio {site}")

    # ===========================
    # LEER CSV FULL TIMELINE
    # ===========================
    df = pd.read_csv(
        f"full_timeline_{site}.csv",
        parse_dates=['valid_time']
    )

    # ===========================
    # COLUMNAS BASE (UNÍVOCAS)
    # ===========================
    base = (
        df[['valid_time', 'corrida', 'leadtime', 'ghi', 'GHI_Wm2']]
        .drop_duplicates(
            subset=['valid_time', 'corrida', 'leadtime']
        )
        .sort_values(['valid_time', 'corrida', 'leadtime'])
        .reset_index(drop=True)
    )

    # ===========================
    # PIVOT DE PREDICCIONES
    # ===========================
    pred = (
        df[['valid_time', 'corrida', 'leadtime', 'modelo', 'GHI_adaptada']]
        .pivot(
            index=['valid_time', 'corrida', 'leadtime'],
            columns='modelo',
            values='GHI_adaptada'
        )
        .reset_index()
    )

    # ===========================
    # RENOMBRAR COLUMNAS DE MODELOS
    # ===========================
    pred.columns = [
        c if c in ['valid_time', 'corrida', 'leadtime']
        else f'GHI_MLP_{c}'
        for c in pred.columns
    ]

    # ===========================
    # MERGE FINAL
    # ===========================
    df_final = (
        base.merge(
            pred,
            on=['valid_time', 'corrida', 'leadtime'],
            how='left'
        )
        .sort_values(['valid_time', 'corrida', 'leadtime'])
        .reset_index(drop=True)
    )

    # ===========================
    # GUARDAR
    # ===========================
    output = f"full_timeline_{site}_multi_modelo.csv"
    df_final.to_csv(output, index=False)

    print(f"✔ Generado {output}")
