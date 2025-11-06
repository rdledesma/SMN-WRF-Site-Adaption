import pandas as pd
import Metrics as ms
import matplotlib.pyplot as plt

# --- Observaciones ---
d = pd.read_excel('medidas/BA_2023_QC_60min.xlsx')
d.columns = ['datetime','ghi','sza','GHIcc']
d['datetime'] = pd.to_datetime(d['datetime'])

d = d[d.sza<90]

# --- Modelados ---
dfMod = pd.DataFrame()
for x in ['01','02','03','04','05','06','07','12']:
    mod = pd.read_csv(f'modelados/BA_2023_{x}_all.csv')
    dfMod = pd.concat([dfMod, mod])

from datetime import timedelta
dfMod['valid_time'] = pd.to_datetime(dfMod['valid_time']) - timedelta(minutes=60)




# --- Merge entre modelo y observación ---
df = pd.merge(
    dfMod[['corrida','leadtime','valid_time','GHI_Wm2']],
    d[['datetime','ghi']],
    left_on='valid_time', right_on='datetime', how='inner'
)

df = df.dropna()


# --- Cálculo de métricas por corrida y leadtime ---
results = []
for (corrida, lt), g in df.groupby(['corrida','leadtime']):
    obs = g['ghi'].values
    mod = g['GHI_Wm2'].values

    rmse = ms.rrmsd(mod, obs)
    rmae = ms.rmae(mod, obs)
    rmbe = ms.rmbe(mod, obs)

    results.append({
        'corrida': corrida,
        'leadtime': lt,
        'RMSE': rmse,
        'RMAE': rmae,
        'RMBE': rmbe
    })

dfMetrics = pd.DataFrame(results)


plt.figure(figsize=(10,6))

for corrida, g in dfMetrics.groupby('corrida'):
    plt.plot(g['leadtime'], g['RMSE'], '.-',label=f'Corrida {corrida}',)

plt.xlabel('Leadtime (h)')
plt.ylabel('RMSE')
plt.title('Degradación del modelo de GHI (RMSE) por corrida')
plt.legend(ncol=2, fontsize='small')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
