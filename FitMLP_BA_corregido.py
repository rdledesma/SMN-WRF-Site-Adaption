# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, train_test_split
import joblib
import glob
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import Metrics as ms  # tus métricas personalizadas (rrmsd, rmbe)

np.random.seed(42)

# ---------------------------
#  Config
# ---------------------------
SITE = 'BA'
MODEL_PREFIX = f"MLP_{SITE}_best"
SCALER_X_NAME = f"scaler_X_{SITE}.pkl"
SCALER_y_NAME = f"scaler_y_{SITE}.pkl"
MAX_EPOCHS = 500
BATCH_SIZE = None  # None => usar todo el X_train (puedes implementar mini-batches si quieres)
# ---------------------------

# === Cargar datos ===
archivos = glob.glob(f"modelados/{SITE}*.csv")
df_all = pd.concat([pd.read_csv(a) for a in archivos], ignore_index=True)
df_all['valid_time'] = pd.to_datetime(df_all.valid_time) - timedelta(minutes=60)

archivosMedidos = glob.glob(f"medidas/{SITE}*.xlsx")
d = pd.concat([pd.read_excel(a) for a in archivosMedidos], ignore_index=True)
d.columns = ['datetime','ghi','sza','GHIcc']
d['datetime'] = pd.to_datetime(d.datetime)

df = pd.merge(
    df_all,
    d[['datetime','ghi','sza']],
    left_on='valid_time',
    right_on='datetime',
    how='inner'
).dropna()
df = df[df.sza < 90].reset_index(drop=True)

# === Variables ===
features = ['T2','TSLB','GHI_Wm2']
target = 'ghi'

X_test = df[df.datetime.dt.year != 2023][features].copy()
y_test = df[df.datetime.dt.year != 2023][target].copy().values.reshape(-1,1)  # mantener como columna para scaler_y


X = df[df.datetime.dt.year == 2023][features].copy()
y = df[df.datetime.dt.year == 2023][target].copy().values.reshape(-1,1)  # mantener como columna para scaler_y


# === Splits (igual que tu original, pero con comprobaciones) ===
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, shuffle=True
)


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)


print(f"Train: {len(X_train)/len(df):.1%}  ({len(X_train)} muestras)")
print(f"Validation: {len(X_val)/len(df):.1%}  ({len(X_val)} muestras)")
print(f"Test: {len(X_test)/len(df):.1%}  ({len(X_test)} muestras)")

# === Escalado (IMPORTANTE: escalo X y también y) ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
X_val_s = scaler_X.transform(X_val)
X_test_s = scaler_X.transform(X_test)

y_train_s = scaler_y.fit_transform(y_train).ravel()  # a 1D para sklearn
y_val_s = scaler_y.transform(y_val).ravel()
y_test_s = scaler_y.transform(y_test).ravel()

# === Grid de hiperparámetros ===
param_grid = {
    'hidden_layer_sizes': [(50,20,10), (50,20)],
    'activation': ['relu'],
    'learning_rate_init': [ 0.001, 0.0005],
    'solver': ['adam'],  # adam funciona bien; podrías probar 'sgd' si haces mini-batches
    'alpha': [1e-4, 1e-3]  # regularización L2
}

# Para registrar resultados
resultados = []

mejor_modelo = None
mejor_params = None
mejor_rrmsd = np.inf
mejor_rmbe = np.inf

# === Entrenamiento por grid: usamos partial_fit para tener control por época ===
for params in ParameterGrid(param_grid):
    print(f"\nEntrenando con: {params}")

    # Configuro el MLP para entrenamiento incremental con partial_fit
    mlp = MLPRegressor(random_state=42,
                       warm_start=True,  # partial_fit no necesita warm_start pero lo dejamos por si acaso
                       max_iter=1,        # no se usa directamente: usaremos partial_fit
                       **params)

    # Inicializo pesos llamando a partial_fit una vez (necesario para algunos backends)
    # WARNING: partial_fit para regresión está soportado; lo usamos para simular epochs.
    train_losses = []
    val_losses = []

    # Algunos MLP requieren una primera llamada a fit/partial_fit para inicializar capas.
    # Llamamos partial_fit repetidamente durante MAX_EPOCHS
    for epoch in range(MAX_EPOCHS):
        # si quieres mini-batches podrías iterar sobre particiones de X_train_s aquí
        mlp.partial_fit(X_train_s, y_train_s)

        # Predicciones (en escala de trabajo)
        y_tr_pred_s = mlp.predict(X_train_s)
        y_val_pred_s = mlp.predict(X_val_s)

        # calcular MSE en escala estandarizada (para seguimiento) o desescalar si prefieres en unidades originales
        mse_tr = np.mean((y_train_s - y_tr_pred_s)**2)
        mse_val = np.mean((y_val_s - y_val_pred_s)**2)

        train_losses.append(mse_tr)
        val_losses.append(mse_val)

        # criterio simple de parada temprana (manual)
        if epoch > 100:
            # si la validación no mejoró en 30 epochs -> break
            if np.argmin(val_losses[:-100]) == len(val_losses) - 31:
                # la mejor validación fue hace 30 epochs -> no mejoró
                break

    # Una vez entrenado en escala, predecimos en escala original desescalando
    y_val_pred_original = scaler_y.inverse_transform(y_val_pred_s.reshape(-1,1)).ravel()
    y_val_original = y_val.ravel()

    # Tus métricas personalizadas (las calculamos en unidades originales)
    rrmsd_val = ms.rrmsd(y_val_original, y_val_pred_original)
    rmbe_val = ms.rmbe(y_val_original, y_val_pred_original)

    resultados.append({
        'params': params,
        'rrmsd': rrmsd_val,
        'rmbe': rmbe_val,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': len(train_losses),
        'modelo': mlp
    })

    print(f" --> rrmsd(val) = {rrmsd_val:.4f}, rmbe(val) = {rmbe_val:.4f}, epochs={len(train_losses)}")

    # Selección del mejor: prioridad rrmsd (tie-breaker rmbe)
    if rrmsd_val < mejor_rrmsd or (np.isclose(rrmsd_val, mejor_rrmsd) and abs(rmbe_val) < abs(mejor_rmbe)):
        mejor_rrmsd = rrmsd_val
        mejor_rmbe = rmbe_val
        mejor_modelo = mlp
        mejor_params = params
        mejor_train_losses = train_losses
        mejor_val_losses = val_losses

    # Graficar la curva de pérdidas para esta configuración (opcional)
    plt.figure(figsize=(7,4))
    plt.plot(train_losses, label='train MSE (scaled)')
    plt.plot(val_losses, label='val MSE (scaled)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (scaled)')
    plt.title(f"Loss curve params: {params}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=False)

# === Resultados ===
print("\n=== Mejor modelo según rrmsd en validación ===")
print(mejor_params)
print(f"rrmsd={mejor_rrmsd:.4f}, rmbe={mejor_rmbe:.4f}")

# Guardar scalers y modelo
joblib.dump(scaler_X, SCALER_X_NAME)
joblib.dump(scaler_y, SCALER_y_NAME)
joblib.dump(mejor_modelo, MODEL_PREFIX + ".pkl")
print(f"Guardado: {SCALER_X_NAME}, {SCALER_y_NAME}, {MODEL_PREFIX}.pkl")

# === Evaluación final en Test ===
# recuperar muestras originales (ya teníamos X_test, y_test)
y_test_pred_s = mejor_modelo.predict(X_test_s)
y_test_pred = scaler_y.inverse_transform(y_test_pred_s.reshape(-1,1)).ravel()

y_test_orig = y_test.ravel()
rmse_test = ms.rrmsd(y_test_orig, y_test_pred)
rmbe_test = ms.rmbe(y_test_orig, y_test_pred)

# comparar con modelo original (WRF)
test_df = df.loc[X_test.index].copy()
rmse_test_orig = ms.rrmsd(y_test_orig, test_df['GHI_Wm2'].values)
rmbe_test_orig = ms.rmbe(y_test_orig, test_df['GHI_Wm2'].values)

print(f"Test - rrmsd={rmse_test:.4f}, rmbe={rmbe_test:.4f}")
print(f"WRF original Test - rrmsd={rmse_test_orig:.4f}, rmbe={rmbe_test_orig:.4f}")

# === Graficar comparativa test (primeras 500 muestras) ===
nplot = min(500, len(y_test_orig))
plt.figure(figsize=(10,5))
plt.plot(y_test_orig[:nplot], label='Observado', linewidth=1)
plt.plot(y_test_pred[:nplot], label=f'MLP predicción (rRMSE={rmse_test:.3f})', linestyle='--', alpha=0.8)
plt.plot(test_df['GHI_Wm2'].values[:nplot], label=f'WRF original (rRMSE={rmse_test_orig:.3f})', linestyle=':', alpha=0.7)
plt.legend()
plt.xlabel('Número de muestra')
plt.ylabel('GHI [W/m²]')
plt.title(f'Comparación en conjunto Test - {SITE}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
