import pandas as pd
import Metrics as ms
import matplotlib.pyplot as plt
import glob
from datetime import timedelta

SITE = 'BA'

archivosMedidos = glob.glob(f"medidas/{SITE}*.xlsx")
d = pd.concat(
    [pd.read_excel(archivo) for archivo in archivosMedidos],
    ignore_index=True
)

d.columns = ['datetime','ghi','sza','GHIcc']
d['datetime'] = pd.to_datetime(d.datetime)


plt.figure()
plt.plot(d.datetime, d.ghi, '.-r')
plt.show()


archivos = glob.glob(f"modelados/{SITE}*.csv")

dfMod = pd.DataFrame()

# Leer y concatenar todos los archivos encontrados
dfMod = pd.concat(
    [pd.read_csv(archivo) for archivo in archivos],
    ignore_index=True
)

#debido a que la etiqueta toporal de WRF indica en fin del intervalo
dfMod['valid_time'] = pd.to_datetime(dfMod.valid_time) - timedelta(minutes = 60)

df = pd.merge(dfMod[['corrida','leadtime','valid_time','GHI_Wm2']],
              d[['datetime','ghi','sza']],
              left_on='valid_time', right_on='datetime', how='inner')

mask_corrida_0 = df.corrida ==0
mask_corrida_12 = df.corrida ==12

plt.figure()
plt.plot(df[mask_corrida_0].valid_time, df[mask_corrida_0].GHI_Wm2,  '.b', label="GHI WRF" )
plt.plot(df[mask_corrida_0].valid_time, df[mask_corrida_0].ghi,  '-.r', label="Medida", )

plt.plot(df[mask_corrida_12].valid_time, df[mask_corrida_12].GHI_Wm2,  '.g', label="GHI WRF" )
plt.plot(df[mask_corrida_12].valid_time, df[mask_corrida_12].ghi,  '*c', label="Medida", )
plt.show()


df = df.dropna()


results = []

for lt, g in df.groupby('leadtime'):
    obs = g['ghi'].values
    mod = g['GHI_Wm2'].values
    hora = g['valid_time'].dt.hour.values[0]

    rmse = ms.rrmsd(mod, obs)
    rmae = ms.rmae(mod, obs)
    rmbe = ms.rmbe(mod, obs)
    

    results.append({'leadtime': lt,  'RMBE': rmbe, 'RMAE': rmae,'RMSE': rmse})

dfMetrics = pd.DataFrame(results)


import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(dfMetrics['leadtime'], dfMetrics['RMSE'], label='RMSE', marker='o')
plt.plot(dfMetrics['leadtime'], dfMetrics['RMAE'], label='RMAE', marker='s')
plt.plot(dfMetrics['leadtime'], dfMetrics['RMBE'], label='RMBE', marker='^')

plt.xlabel('(h) Cada una de las 72 horas de pronóstico')
plt.ylabel('Error')
plt.title('Degradación del modelo de GHI según horizonte de pronóstico en Buenos Aires')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()





# Filtrar horas diurnas (solo cuando hay radiación medida significativa)
df_day = df[df['sza'] <90 ]  # o df[df['sza'] < 90] si tienes ese dato
 
results_day = []
for lt, g in df_day.groupby('leadtime'):
    obs = g['ghi'].values
    mod = g['GHI_Wm2'].values
    rmse = ms.rrmsd(mod, obs)
    rmae = ms.rmae(mod, obs)
    rmbe = ms.rmbe(mod, obs)
    results_day.append({'leadtime': lt, 'RMBE': rmbe, 'RMAE': rmae, 'RMSE': rmse})

dfMetrics_day = pd.DataFrame(results_day)

plt.figure(figsize=(10,6))
plt.plot(dfMetrics_day['leadtime'], dfMetrics_day['RMSE'], label='RMSE', marker='o')
plt.plot(dfMetrics_day['leadtime'], dfMetrics_day['RMAE'], label='RMAE', marker='s')
plt.plot(dfMetrics_day['leadtime'], dfMetrics_day['RMBE'], label='RMBE', marker='^')
plt.xlabel('Lead time (h)')
plt.ylabel('Error')
plt.title('Degradación del modelo WRF GHI (solo horas diurnas)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()





import numpy as np

df_day['error'] = df_day['GHI_Wm2'] - df_day['ghi']

plt.figure(figsize=(12,6))
df_day.boxplot(column='error', by='leadtime', grid=False, showfliers=False)
plt.title('Distribución del error GHI (modelado - observado) por lead time')
plt.suptitle('')
plt.xlabel('Lead time (h)')
plt.ylabel('Error (W/m²)')
plt.grid(True, alpha=0.3)
plt.show()




import seaborn as sns

df_day['hora'] = df_day['valid_time'].dt.hour
heat_data = df_day.groupby(['hora','leadtime'])['GHI_Wm2'].mean() - df_day.groupby(['hora','leadtime'])['ghi'].mean()
heat_data = heat_data.unstack()

plt.figure(figsize=(12,6))
sns.heatmap(heat_data, cmap='RdBu_r', center=0)
plt.title('Error medio (modelado - observado) por hora local y lead time')
plt.xlabel('Lead time (h)')
plt.ylabel('Hora local')
plt.show()




import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

for i, (min_lt, max_lt, color, label) in enumerate([
    (0, 12, 'blue', '0–12h'),
    (13, 24, 'green', '13–24h'),
    (25, 48, 'orange', '25–48h'),
    (49, 72, 'red', '49–72h')
]):
    subset = df_day[(df_day.leadtime >= min_lt) & (df_day.leadtime <= max_lt)]
    plt.scatter(subset['ghi'], subset['GHI_Wm2'], alpha=0.3, s=10, label=label, color=color)

plt.plot([0, 1000], [0, 1000], 'k--')  # línea 1:1
plt.xlabel('Observado (W/m²)')
plt.ylabel('Modelado (W/m²)')
plt.title('Comparación Observado vs Modelado para distintos horizontes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
