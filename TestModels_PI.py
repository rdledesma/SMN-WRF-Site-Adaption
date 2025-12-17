# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import glob
import datetime
from datetime import timedelta
import Metrics as ms  # tus métricas personalizadas (rrmsd, rmbe)
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
#  Configuración
# ---------------------------
SITE = 'PI'
MODEL_PREFIX = f"MLP_{SITE}_best"
SCALER_X_NAME = f"scaler_X_{SITE}.pkl"
SCALER_y_NAME = f"scaler_y_{SITE}.pkl"
DAY_OBS = datetime.date(2023, 6, 1)  # fecha de análisis
CORRIDA = 0                          # número de corrida
LEAD_MAX = 24                        # horas
FEATURES = ['T2', 'TSLB', 'GHI_Wm2']
# ---------------------------

# === Cargar datos ===
archivos = glob.glob(f"modelados/{SITE}*.csv")
df_all = pd.concat([pd.read_csv(a) for a in archivos], ignore_index=True)
df_all['valid_time'] = pd.to_datetime(df_all.valid_time) - timedelta(minutes=180+60)

archivosMedidos = glob.glob(f"medidas/{SITE}*.xlsx")
d = pd.concat([pd.read_excel(a) for a in archivosMedidos], ignore_index=True)
d.columns = ['datetime', 'ghi', 'sza', 'GHIcc']
d['datetime'] = pd.to_datetime(d.datetime)

# === Merge y filtrado ===
df = pd.merge(
    df_all,
    d[['datetime', 'ghi', 'sza']],
    left_on='valid_time',
    right_on='datetime',
    how='inner'
)
#df = df[df.datetime.dt.date < DAY_OBS]
df = df[df.corrida == CORRIDA]
df = df[df.leadtime < LEAD_MAX]
df = df.sort_values(by=['datetime']).reset_index(drop=True)

if df.empty:
    raise ValueError("No hay datos que cumplan los filtros establecidos.")

# === Preparar features ===
X = df[FEATURES].interpolate()

# === Cargar modelos y escaladores ===
scaler_X = joblib.load(SCALER_X_NAME)
scaler_y = joblib.load(SCALER_y_NAME)
model = joblib.load(f"{MODEL_PREFIX}.pkl")

# === Predicciones ===
X_scaled = scaler_X.transform(X)
y_pred_s = model.predict(X_scaled).reshape(-1, 1)
y_pred = scaler_y.inverse_transform(y_pred_s).ravel()


df['y_pred'] = y_pred
# === Métricas ===
rrmsd_wrf = ms.rrmsd(df.dropna().ghi, df.dropna().GHI_Wm2)
rrmsd_mlp = ms.rrmsd(df.dropna().ghi, df.dropna().y_pred)
rmbe_wrf = ms.rmbe(df.dropna().ghi, df.dropna().GHI_Wm2)
rmbe_mlp = ms.rmbe(df.dropna().ghi, df.dropna().y_pred)

# === Errores ===
err_wrf = df.GHI_Wm2 - df.ghi
err_mlp = y_pred - df.ghi

# === Gráfico descriptivo ===
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                        gridspec_kw={'height_ratios': [3, 1]})

# --- GHI observado y predicho ---
axs[0].plot(df.datetime, df.ghi, label='Observado', color='black', linewidth=1.5)
axs[0].plot(df.datetime, df.GHI_Wm2.interpolate(), label=f'WRF (rRMSE={rrmsd_wrf:.3f}, rMBE={rmbe_wrf:.3f})',
            linestyle=':', color='tab:blue', linewidth=1.2)
axs[0].plot(df.datetime, y_pred, label=f'WRF+MLP (rRMSE={rrmsd_mlp:.3f}, rMBE={rmbe_mlp:.3f})',
            linestyle='--', color='tab:red', linewidth=1.3)

axs[0].set_ylabel('GHI [W/m²]', fontsize=11)
axs[0].set_title(
    f'Corrección de GHI para {SITE} - Corrida {CORRIDA} (<{LEAD_MAX}h)\n'
    f'Período: 2023-06-01 / 2023-06-02',    
    fontsize=13, weight='bold'
)
axs[0].legend(frameon=False, loc='upper left')
axs[0].grid(alpha=0.3)

# --- Errores ---
axs[1].plot(df.datetime, err_wrf, label='Error WRF', color='tab:blue', linestyle=':')
axs[1].plot(df.datetime, err_mlp, label='Error WRF+MLP', color='tab:red', linestyle='--')
axs[1].axhline(0, color='k', linewidth=0.8)
axs[1].set_ylabel('Error [W/m²]', fontsize=11)
axs[1].set_xlabel('Tiempo (UTC-3)', fontsize=11)
axs[1].legend(frameon=False, loc='upper left')
axs[1].grid(alpha=0.3)

# --- Ejes de tiempo formateados ---
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
axs[1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=30, ha='right')

# --- Información extra dentro del gráfico ---
info_text = (
    f"Sitio: {SITE}\n"
    f"Corrida: {CORRIDA}\n"
    f"Leadtime  {LEAD_MAX}h\n"
    f"Muestras: {len(df)}\n"
    f"WRF  → rRMSE={rrmsd_wrf:.3f}, rMBE={rmbe_wrf:.3f}\n"
    f"MLP  → rRMSE={rrmsd_mlp:.3f}, rMBE={rmbe_mlp:.3f}"
)
axs[0].text(1.02, 0.45, info_text, transform=axs[0].transAxes,
            fontsize=10, va='center', ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.tight_layout()
plt.subplots_adjust(right=0.83)
plt.show()
