# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import glob
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")

# ===========================
# CONFIGURACIÓN GENERAL
# ===========================
SITES = ['AR', 'PI', 'BA']
FEATURES = ['T2', 'TSLB', 'GHI_Wm2']

MODEL_PREFIX = "MLP_{}_best"
SCALER_X = "scaler_X_{}.pkl"
SCALER_y = "scaler_y_{}.pkl"

LEAD_MAX = 73


def get_time_shift(site):
    return timedelta(minutes=60 if site == 'BA' else 240)


# ===========================
# FUNCIÓN PRINCIPAL
# ===========================
def generar_full_timeline(site):
    print(f"\nProcesando sitio {site}")

    TIME_SHIFT = get_time_shift(site)

    # ===========================
    # DATOS MODELADOS (BASE ABSOLUTA)
    # ===========================
    archivos = glob.glob(f"modelados/{site}*.csv")
    df_model = pd.concat(
        [pd.read_csv(a) for a in archivos],
        ignore_index=True
    )

    # Ajuste horario
    df_model['valid_time'] = (
        pd.to_datetime(df_model['valid_time']) - TIME_SHIFT
    )

    # Filtrado SOLO por leadtime (no se toca el orden)
    df_model = df_model[df_model.leadtime < LEAD_MAX].reset_index(drop=True)

    # ===========================
    # DATOS OBSERVADOS
    # ===========================
    archivos_med = glob.glob(f"medidas/{site}*.xlsx")

    if archivos_med:
        df_obs = pd.concat(
            [pd.read_excel(a) for a in archivos_med],
            ignore_index=True
        )
        df_obs.columns = ['datetime', 'ghi', 'sza', 'GHIcc']
        df_obs['datetime'] = pd.to_datetime(df_obs['datetime'])
    else:
        df_obs = pd.DataFrame(columns=['datetime', 'ghi'])

    # Merge observaciones (NO altera estructura)
    df = df_model.merge(
        df_obs[['datetime', 'ghi']],
        left_on='valid_time',
        right_on='datetime',
        how='left'
    ).drop(columns=['datetime'])

    # ===========================
    # FEATURES (MISMO ORDEN)
    # ===========================
    X = df[FEATURES].interpolate(limit_direction='both')

    # ===========================
    # PREDICCIONES (MISMA FILA = MISMO LEAD)
    # ===========================
    for model_site in SITES:
        print(f"  → aplicando modelo {model_site}")

        model = joblib.load(MODEL_PREFIX.format(model_site) + ".pkl")
        scaler_X = joblib.load(SCALER_X.format(model_site))
        scaler_y = joblib.load(SCALER_y.format(model_site))

        X_scaled = scaler_X.transform(X)
        y_scaled = model.predict(X_scaled).reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_scaled).ravel()

        df[f'GHI_MLP_{model_site}'] = y_pred

    # ===========================
    # GUARDAR (FORMATO ORIGINAL + NUEVAS COLUMNAS)
    # ===========================
    output = f"full_timeline_{site}_multi_modelo.csv"

    df = df[['date','corrida','leadtime','valid_time','GHI_Wm2','ghi','GHI_MLP_AR','GHI_MLP_PI','GHI_MLP_BA']]
    df['valid_time'] = df.valid_time +  TIME_SHIFT

    df = (
    df.sort_values(['date', 'corrida', 'leadtime'])
      .reset_index(drop=True))


    df.to_csv(output, index=False)

    print(f"✔ Generado {output}")


# ===========================
# EJECUCIÓN
# ===========================
for site in SITES:
    generar_full_timeline(site)
