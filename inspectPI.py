import pandas as pd
import Metrics as ms
import matplotlib.pyplot as plt
d = pd.read_excel('medidas/PI_2023_QC_60min.xlsx')
d.columns = ['datetime','ghi','sza','GHIcc']
d['datetime'] = pd.to_datetime(d.datetime)


plt.figure()
plt.plot(d.datetime, d.ghi, '-r')
plt.show()



dfMod = pd.DataFrame()
for x in ['01','02','12']:
    mod = pd.read_csv(f'modelados/PI_2023_{x}_all.csv')
    dfMod = pd.concat([dfMod,mod])

from datetime import timedelta
dfMod['valid_time'] = pd.to_datetime(dfMod.valid_time) - timedelta(minutes = 180+60)

df = pd.merge(dfMod[['corrida','leadtime','valid_time','GHI_Wm2']],
              d[['datetime','ghi']],
              left_on='valid_time', right_on='datetime', how='inner')


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
plt.title('Degradación del modelo de GHI según horizonte de pronóstico en El Pilar')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

