import pandas as pd
import numpy as np
site = 'PI'
year = '2023'
for month in  ['03']:

    df = pd.read_csv(f'modelados/raw/{site}_PP_HR2_T2_dirViento10_magViento10_PSFC_ACLWDNB_ACLWUPB_ACSWDNB_TSLB_SMOIS_Freezing_level_Tmax_Tmin_{year}_{month}.csv')

    var = 'ACSWDNB'
    df["delta_Jm2"] = df.groupby(["date", "corrida"])[var].diff()
    df["dt_seconds"] = 3600  # porque usamos 01H
    df["GHI_Wm2"] = df["delta_Jm2"] / df["dt_seconds"]

    # Manejar NaN en el primer lead de cada corrida
    df.loc[df["leadtime"] == 0, "GHI_Wm2"] = np.nan

    print(df.head())

    df.columns
    df = df[['date', 'corrida', 'leadtime', 'valid_time', 'PP', 'HR2', 'T2',
        'dirViento10', 'magViento10', 'PSFC', 'ACLWDNB', 'ACLWUPB', 'ACSWDNB',
        'TSLB', 'SMOIS',  'GHI_Wm2']]


    # Guardar CSV
    csv_file = f"{site}_{year}_{month}_all.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSerie guardada en {csv_file}")