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
import seaborn as sns  # Asegurándonos de que seaborn esté importado

warnings.filterwarnings("ignore")

# ---------------------------
#  Configuración
# ---------------------------
SITE = 'BA'
MODEL_PREFIX = f"MLP_{SITE}_best"
SCALER_X_NAME = f"scaler_X_{SITE}.pkl"
SCALER_y_NAME = f"scaler_y_{SITE}.pkl"
DAY_OBS = datetime.date(2023, 6, 1)  # fecha de análisis
CORRIDA = 0                          # número de corrida
LEAD_MAX = 24                        # horas
FEATURES = ['PP', 'HR2', 'T2',
       'dirViento10', 'magViento10', 'PSFC', 'ACLWDNB', 'ACLWUPB', 'ACSWDNB',
       'TSLB', 'SMOIS', 'GHI_Wm2']
# ---------------------------

# === Cargar datos ===
archivos = glob.glob(f"modelados/{SITE}*.csv")
df_all = pd.concat([pd.read_csv(a) for a in archivos], ignore_index=True)
df_all['valid_time'] = pd.to_datetime(df_all.valid_time) - timedelta(minutes=60)

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


# 1. Correlación de Pearson
pearson_corr = df[['ghi', 'GHI_Wm2',  'PP', 'HR2', 'T2',
       'dirViento10', 'magViento10', 'PSFC', 'ACLWDNB', 'ACLWUPB', 'ACSWDNB',
       'TSLB', 'SMOIS']].corr(method='pearson')

# Extraemos las correlaciones de GHI con las demás variables
ghi_corr = pearson_corr['ghi'].drop('ghi')  # Excluimos la correlación de GHI consigo mismo

# Gráfico de barras para la correlación de GHI con otras variables
plt.figure(figsize=(10, 6))
ghi_corr.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title(f'Correlación de GHI medida en {SITE}', fontsize=14)
plt.xlabel('Variables', fontsize=12)
plt.ylabel('Correlación de Pearson', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
