import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import joblib
import Metrics as ms

# === 1. Cargar datos PI ===
df_all = pd.DataFrame()
for x in ['01','02','12']:
    mod = pd.read_csv(f'modelados/PI_2023_{x}_all.csv')
    mod['month'] = x
    df_all = pd.concat([df_all, mod])

df_all['valid_time'] = pd.to_datetime(df_all.valid_time) - timedelta(minutes=180 + 60)

d = pd.read_excel('medidas/PI_2023_QC_60min.xlsx')
d.columns = ['datetime','ghi','sza','GHIcc']
d['datetime'] = pd.to_datetime(d.datetime)

df = pd.merge(
    df_all,
    d[['datetime','ghi']],
    left_on='valid_time',
    right_on='datetime',
    how='inner'
).dropna()

# === 2. Variables ===
features =  ['T2','PSFC', 'ACLWDNB', 'ACLWUPB', 'ACSWDNB', 'TSLB', 'SMOIS','GHI_Wm2']
target = 'ghi'

X = df[features]
y = df[target]

# === 3. Cargar scaler y modelo entrenados con BA ===
scaler = joblib.load('scaler_MLP_BA2023.pkl')
mlp = joblib.load('MLP_BA2023_best.pkl')

# === 4. Transformar y predecir ===
X_scaled = scaler.transform(X)
y_pred = mlp.predict(X_scaled)

# === 5. Evaluar desempeño ===
rrmsd_val = ms.rrmsd(y, y_pred)
rmbe_val = ms.rmbe(y, y_pred)

rmbe_test_original = ms.rmbe(y, X.GHI_Wm2.values)
rrmsd_test_original = ms.rrmsd(y, X.GHI_Wm2.values)
print(f"Test - rrmsd={rrmsd_test_original:.3f}, rmbe={rmbe_test_original:.3f}")




print(f"Resultados PI_2023:")
print(f" → rrmsd = {rrmsd_val:.3f}")
print(f" → rmbe  = {rmbe_val:.3f}")

# === 6. Graficar comparación ===
plt.figure(figsize=(10,5))
plt.plot(y.values, label='Observado', color='k')
plt.plot(y_pred, label=f'Pronosticado + MLP (BA→PI) rRMSE  {rrmsd_val:.2f}', color='r', alpha=0.7)
plt.plot(X.GHI_Wm2.values, label=f'Pronosticado original rRMSE {rrmsd_test_original:.2f}', color='b', alpha=0.6)
plt.legend()
plt.xlabel('Número de muestras')
plt.title('Comparación GHI Observado vs MLP Predicho (PI_2023, una corrida de 72 hs)')
plt.show()

# === 7. Guardar resultados opcionalmente ===
df['GHI_pred_MLP'] = y_pred
df[['datetime','ghi','GHI_pred_MLP']].to_csv('Resultados_MLP_PI2023.csv', index=False)
print("Predicciones guardadas en 'Resultados_MLP_PI2023.csv'")





# --- 1. Calcular métricas por leadtime para GHI_Wm2 y MLP ---
results = []

for lt, g in df.groupby('leadtime'):
    obs = g['ghi'].values
    mod_base = g['GHI_Wm2'].values       # modelo original
    mod_mlp  = g['GHI_pred_MLP'].values  # modelo ajustado con MLP

    rmse_base = ms.rrmsd(mod_base, obs)
    rmae_base = ms.rmae(mod_base, obs)
    rmbe_base = ms.rmbe(mod_base, obs)

    rmse_mlp = ms.rrmsd(mod_mlp, obs)
    rmae_mlp = ms.rmae(mod_mlp, obs)
    rmbe_mlp = ms.rmbe(mod_mlp, obs)

    results.append({
        'leadtime': lt,
        'RMSE_base': rmse_base, 'RMAE_base': rmae_base, 'RMBE_base': rmbe_base,
        'RMSE_mlp': rmse_mlp,  'RMAE_mlp': rmae_mlp,  'RMBE_mlp': rmbe_mlp
    })

dfMetrics = pd.DataFrame(results)

# --- 2. Gráficos comparativos ---
plt.figure(figsize=(10,6))

plt.plot(dfMetrics['leadtime'], dfMetrics['RMSE_base'], label='RMSE Modelo Base', marker='o', color='gray')
plt.plot(dfMetrics['leadtime'], dfMetrics['RMSE_mlp'], label='RMSE MLP Ajustado', marker='o', color='red')

#plt.plot(dfMetrics['leadtime'], dfMetrics['RMAE_base'], label='RMAE Modelo Base', marker='s', color='lightgray')
#plt.plot(dfMetrics['leadtime'], dfMetrics['RMAE_mlp'], label='RMAE MLP Ajustado', marker='s', color='orange')

plt.plot(dfMetrics['leadtime'], dfMetrics['RMBE_base'], label='RMBE Modelo Base', marker='^', color='silver')
plt.plot(dfMetrics['leadtime'], dfMetrics['RMBE_mlp'], label='RMBE MLP Ajustado', marker='^', color='green')

plt.xlabel('(h) Cada una de las 72 horas de pronóstico')
plt.ylabel('Error')
plt.title('Degradación del modelo GHI según horizonte de pronóstico en El Pilar')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()