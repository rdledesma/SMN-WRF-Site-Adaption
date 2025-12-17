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
