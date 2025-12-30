import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import glob
from datetime import timedelta
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

LEAD_MAX = 72


def get_time_shift(site):
    if site == 'BA':
        return timedelta(minutes=60)
    else:
        return timedelta(minutes=240)


# ===========================
# FUNCIÓN PRINCIPAL
# ===========================
def evaluar_y_generar(model_site, val_site):
    print(f"Modelo {model_site} → Validación {val_site}")

    TIME_SHIFT = get_time_shift(val_site)

    # ----- cargar modelo -----
    model = joblib.load(MODEL_PREFIX.format(model_site) + ".pkl")
    scaler_X = joblib.load(SCALER_X.format(model_site))
    scaler_y = joblib.load(SCALER_y.format(model_site))

    # ===========================
    # DATOS MODELADOS (FULL)
    # ===========================
    archivos = glob.glob(f"modelados/{val_site}*.csv")
    df_model = pd.concat(
        [pd.read_csv(a) for a in archivos],
        ignore_index=True
    )

    df_model['valid_time'] = (
        pd.to_datetime(df_model.valid_time) - TIME_SHIFT
    )

    # 🔴 IMPORTANTE: TODAS LAS CORRIDAS
    df_model = df_model[
        df_model.leadtime < LEAD_MAX
    ].sort_values(['corrida', 'valid_time']).reset_index(drop=True)

    # ===========================
    # FEATURES
    # ===========================
    X = df_model[FEATURES].interpolate(limit_direction='both')

    X_scaled = scaler_X.transform(X)
    y_pred_scaled = model.predict(X_scaled).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()

    # ===========================
    # DATOS OBSERVADOS (OPCIONAL)
    # ===========================
    archivos_med = glob.glob(f"medidas/{val_site}*.xlsx")

    if archivos_med:
        df_obs = pd.concat(
            [pd.read_excel(a) for a in archivos_med],
            ignore_index=True
        )
        df_obs.columns = ['datetime', 'ghi', 'sza', 'GHIcc']
        df_obs['datetime'] = pd.to_datetime(df_obs.datetime)
    else:
        df_obs = pd.DataFrame(columns=['datetime', 'ghi'])

    # ===========================
    # MERGE FULL TIMELINE
    # ===========================
    df_out = pd.merge(
        df_model[['valid_time', 'GHI_Wm2', 'corrida', 'leadtime']],
        df_obs[['datetime', 'ghi']],
        left_on='valid_time',
        right_on='datetime',
        how='left'
    )

    df_out.drop(columns=['datetime'], inplace=True)
    df_out['GHI_adaptada'] = y_pred
    df_out['modelo'] = model_site

    # ===========================
    # RMSE (SOLO DONDE HAY OBS)
    # ===========================
    df_eval = df_out.dropna(subset=['ghi'])

    if df_eval.empty:
        rmse = np.nan
    else:
        rmse = ms.rrmsd(df_eval['ghi'], df_eval['GHI_adaptada'])

    return rmse, df_out


# ===========================
# VALIDACIÓN CRUZADA
# ===========================
rmse_matrix = pd.DataFrame(
    index=[f"Modelo_{s}" for s in SITES],
    columns=SITES,
    dtype=float
)

resultados_por_sitio = {s: [] for s in SITES}

for model_site in SITES:
    for val_site in SITES:
        rmse, df_site = evaluar_y_generar(model_site, val_site)
        rmse_matrix.loc[f"Modelo_{model_site}", val_site] = rmse
        resultados_por_sitio[val_site].append(df_site)

print("\nMatriz RMSE:")
print(rmse_matrix)

# ===========================
# GUARDAR CSV FULL TIMELINE
# ===========================
for site, dfs in resultados_por_sitio.items():
    df_final = (
        pd.concat(dfs)
        .sort_values(['corrida', 'valid_time'])
        .reset_index(drop=True)
    )

    df_final.to_csv(
        f"full_timeline_{site}.csv",
        index=False
    )

    print(f"CSV generado: full_timeline_{site}.csv")

# ===========================
# HEATMAP
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
