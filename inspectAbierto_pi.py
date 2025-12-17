import pandas as pd
import Metrics as ms
import matplotlib.pyplot as plt
import glob
from datetime import timedelta

SITE = 'PI'

archivosMedidos = glob.glob(f"medidas/{SITE}*.xlsx")
d = pd.concat(
    [pd.read_excel(archivo) for archivo in archivosMedidos],
    ignore_index=True
)

d.columns = ['datetime','ghi','sza','GHIcc']
d['datetime'] = pd.to_datetime(d.datetime)



archivos = glob.glob(f"modelados/{SITE}*.csv")

dfMod = pd.DataFrame()

# Leer y concatenar todos los archivos encontrados
dfMod = pd.concat(
    [pd.read_csv(archivo) for archivo in archivos],
    ignore_index=True
)

#debido a que la etiqueta toporal de WRF indica en fin del intervalo
dfMod['valid_time'] = pd.to_datetime(dfMod.valid_time) - timedelta(minutes = 180+60)



df = pd.merge(dfMod[['corrida','leadtime','valid_time','GHI_Wm2']],
              d[['datetime','ghi','sza']],
              left_on='valid_time', right_on='datetime', how='inner')

mask_corrida_0 = df.corrida ==0
mask_corrida_12 = df.corrida ==12


df = df.dropna()


df['month'] = df['valid_time'].dt.to_period('M').dt.to_timestamp()


results = []

for (corrida, month), g in df.groupby(['corrida', 'month']):
    obs = g['ghi'].values
    mod = g['GHI_Wm2'].values
    
    if len(obs) < 10:   # opcional: evitar meses con pocos datos
        continue
    
    rrms = ms.rrmsd(mod, obs)

    results.append({
        'corrida': corrida,
        'month': month,
        'rRMSE': rrms
    })

df_rRMSE = pd.DataFrame(results)


plt.figure(figsize=(10, 5))

for corrida, g in df_rRMSE.groupby('corrida'):
    if corrida in [0, 12,18]:
        plt.plot(
            g.month,
            g.rRMSE,
            marker='o',
            linewidth=2,
            label=f'corrida {corrida:02d}'
        )

plt.xlabel('Tiempo')
plt.ylabel('rRMSE')
plt.title('Evolución mensual del rRMSE – GHI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Corridas a graficar
corridas_plot = [0, 6 , 12, 18]
df_plot = df_rRMSE[df_rRMSE.corrida.isin(corridas_plot)].copy()

df_plot['corrida'] = (
    df_plot['corrida']
    .astype(int)
    .apply(lambda x: f'corrida {x:02d}')
)

# Orden temporal explícito
df_plot = df_plot.sort_values('month')

# Estilo paper
sns.set_theme(
    context='paper',
    style='ticks',
    font_scale=1.5
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.lineplot(
    data=df_plot,
    x='month',
    y='rRMSE',
    hue='corrida',
    style='corrida',          # ← distinto marcador por corrida
    markers=True,
    dashes=False,
    linewidth=2.5,
    ax=ax
)

for line in ax.lines:
    line.set_markersize(10)   # ← ajustá este valor a gusto


# Etiquetas
ax.set_xlabel('Tiempo')
ax.set_ylabel('rRMSE')
ax.set_title('Evolución mensual del rRMSE en El Pilar GHI', pad=12)

# ✅ TODOS los meses como xticks
months = df_plot['month'].drop_duplicates().sort_values()
ax.set_xticks(months)
ax.set_xticklabels(
    [m.strftime('%b %Y') for m in months],
    rotation=30,
    ha='right'
)

# Grilla sutil
ax.grid(True, linestyle='--', alpha=0.4)

# Bordes limpios
sns.despine()

# Leyenda clara
ax.legend(
    title='',
    frameon=False,
    loc='upper left'
)

plt.tight_layout()
plt.show()
