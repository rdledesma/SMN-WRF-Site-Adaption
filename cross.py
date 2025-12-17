# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import glob
import datetime
from datetime import timedelta
from sklearn.metrics import mean_squared_error
import warnings
import Metrics as ms
warnings.filterwarnings("ignore")

# ===========================
# CONFIGURACIÓN GENERAL
# ===========================
SITES = ['AR', 'PI', 'BA']

FEATURES = ['T2', 'TSLB', 'GHI_Wm2']

MODEL_PREFIX = "MLP_{}_best"
SCALER_X = "scaler_X_{}.pkl"
SCALER_y = "scaler_y_{}.pkl"

CORRIDA = 0
LEAD_MAX = 24
#TIME_SHIFT = timedelta(minutes=180+60)


def get_time_shift(site):
    if site == 'BA':
        return timedelta(minutes=60)
    else:
        return timedelta(minutes=180 + 60)


# ===========================
# FUNCION DE EVALUACIÓN
# ===========================
def evaluar_modelo(model_site, val_site):


    print(f"Evaluando modelo {model_site} → sitio {val_site}")
    TIME_SHIFT = get_time_shift(val_site)
    # ----- cargar modelo y escaladores -----
    model = joblib.load(MODEL_PREFIX.format(model_site) + ".pkl")
    scaler_X = joblib.load(SCALER_X.format(model_site))
    scaler_y = joblib.load(SCALER_y.format(model_site))

    # ----- datos modelados -----
    archivos = glob.glob(f"modelados/{val_site}*.csv")
    df_all = pd.concat([pd.read_csv(a) for a in archivos], ignore_index=True)

    df_all['valid_time'] = (
        pd.to_datetime(df_all.valid_time) - TIME_SHIFT
    )

    # ----- datos observados -----
    archivos_med = glob.glob(f"medidas/{val_site}*.xlsx")
    d = pd.concat([pd.read_excel(a) for a in archivos_med], ignore_index=True)
    d.columns = ['datetime', 'ghi', 'sza', 'GHIcc']
    d['datetime'] = pd.to_datetime(d.datetime)

    # ----- merge -----
    df = pd.merge(
        df_all,
        d[['datetime', 'ghi']],
        left_on='valid_time',
        right_on='datetime',
        how='inner'
    )

    # ----- filtros -----
    df = df[df.corrida == CORRIDA]
    df = df[df.leadtime < LEAD_MAX]
    df = df.sort_values(by='datetime').reset_index(drop=True)
    df = df.dropna(subset=FEATURES + ['ghi'])

    if df.empty:
        print("⚠️ Sin datos válidos")
        return np.nan

    # ----- features -----
    X = df[FEATURES].interpolate()

    # ----- predicción -----
    X_scaled = scaler_X.transform(X)
    y_pred_scaled = model.predict(X_scaled).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()

    # ----- RMSE -----
    rmse = ms.rrmsd(df['ghi'], y_pred)

    return rmse


# ===========================
# MATRIZ DE VALIDACION CRUZADA
# ===========================
rmse_matrix = pd.DataFrame(
    index=[f"Modelo_{s}" for s in SITES],
    columns=SITES,
    dtype=float
)

for model_site in SITES:
    for val_site in SITES:
        rmse_matrix.loc[f"Modelo_{model_site}", val_site] = evaluar_modelo(
            model_site,
            val_site
        )

print("\nMatriz RMSE:")
print(rmse_matrix)

# ===========================
# GRAFICO HEATMAP
# ===========================
plt.figure(figsize=(7, 5))
sns.heatmap(
    rmse_matrix,
    annot=True,
    fmt=".0f",
    cmap="viridis",
    cbar_kws={'label': 'rRMSE [W m$^{-2}$]'}
)

plt.title("Validación cruzada entre sitios – RMSE")
plt.xlabel("Sitio de validación")
plt.ylabel("Modelo entrenado en")
plt.tight_layout()
plt.show()
