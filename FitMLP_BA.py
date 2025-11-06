import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import joblib
import Metrics as ms   # tus métricas personalizadas
import glob
from sklearn.model_selection import train_test_split


# === 1. Cargar datos ===
SITE = 'BA'

archivos = glob.glob(f"modelados/{SITE}*.csv")
dfMod = pd.DataFrame()
# Leer y concatenar todos los archivos encontrados
df_all= pd.concat(
    [pd.read_csv(archivo) for archivo in archivos],
    ignore_index=True
)
df_all['valid_time'] = pd.to_datetime(df_all.valid_time) - timedelta(minutes = 60)


archivosMedidos = glob.glob(f"medidas/{SITE}*.xlsx")
d = pd.concat(
    [pd.read_excel(archivo) for archivo in archivosMedidos],
    ignore_index=True
)


d.columns = ['datetime','ghi','sza','GHIcc']
d['datetime'] = pd.to_datetime(d.datetime)

df = pd.merge(
    df_all,
    d[['datetime','ghi','sza']],
    left_on='valid_time',
    right_on='datetime',
    how='inner'
).dropna()

df  = df[df.sza<90]

# === 2. Variables ===
features = ['T2','TSLB','GHI_Wm2']
target = 'ghi'


X = df[features]
y = df[target]

# === 1️⃣ Primer split: separar test (50%) del resto (50%) ===
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, shuffle=True
)

# === 2️⃣ Segundo split: dividir el 50% restante en 40% train y 10% val ===
# Nota: 10% del total equivale a 1/5 = 20% del grupo train_val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, shuffle=True
)

# === Verificar proporciones ===
print(f"Train: {len(X_train)/len(df):.1%}")
print(f"Validation: {len(X_val)/len(df):.1%}")
print(f"Test: {len(X_test)/len(df):.1%}")




# === 3. Escalado ===
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# === 4. Definimos grilla de hiperparámetros ===
param_grid = {
    'hidden_layer_sizes': [(10,), (20,), (30,), (50,20), (100,50)],
    'activation': ['relu'],
    'learning_rate_init': [ 0.01, 0.001],
    'max_iter': [500]
}

# === 5. Entrenar y evaluar manualmente ===
mejor_modelo = None
mejor_params = None
mejor_rmse = np.inf
mejor_rmbe = np.inf

for params in ParameterGrid(param_grid):
    print(f"Entrenando con: {params}")
    mlp = MLPRegressor(random_state=42, **params)
    mlp.fit(X_train_s, y_train)
    
    # Predicción en validación
    y_pred_val = mlp.predict(X_val_s)
    
    # Métricas personalizadas
    rmse = ms.rrmsd(y_val, y_pred_val)
    rmbe = ms.rmbe(y_val, y_pred_val)
    
    print(f" → rrmsd={rmse:.3f}, rmbe={rmbe:.3f}")
    
    # Seleccionar mejor modelo según rrmsd
    if (rmse < mejor_rmse) and (abs(rmbe) < abs(mejor_rmbe)):
        mejor_rmse = rmse
        mejor_rmbe = rmbe
        mejor_modelo = mlp
        mejor_params = params

print("\n=== Mejor modelo según rrmsd ===")
print(mejor_params)
print(f"rrmsd={mejor_rmse:.3f}, rmbe={mejor_rmbe:.3f}")

# === 6. Guardar scaler y modelo ===
joblib.dump(scaler, 'scaler_MLP_BA2023_all.pkl')
joblib.dump(mejor_modelo, 'MLP_BA2023_best_all.pkl')
print("Modelos guardados: scaler_MLP_BA2023.pkl, MLP_BA2023_best.pkl")

# === 7. Evaluación final en test ===

# Recuperar también la columna original 'GHI_Wm2' del set de test
test = df.loc[X_test.index].copy()

# Predicción con el mejor modelo
y_pred_test = mejor_modelo.predict(X_test_s)

# === Métricas ===
rmse_test = ms.rrmsd(y_test, y_pred_test)
rmbe_test = ms.rmbe(y_test, y_pred_test)

# Comparación con el modelo original (WRF)
rmse_test_original = ms.rrmsd(y_test, test['GHI_Wm2'].values)
rmbe_test_original = ms.rmbe(y_test, test['GHI_Wm2'].values)

# === Resultados en consola ===
print(f"Test - rrmsd={rmse_test:.3f}, rmbe={rmbe_test:.3f}")
print(f"TestOriginal - rrmsd={rmse_test_original:.3f}, rmbe={rmbe_test_original:.3f}")

# === 8. Graficar resultados del test ===
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:500], label='Observado', color='k')
plt.plot(y_pred_test[:500], label=f'MLP predicción (rRMSE={rmse_test:.2f})', color='r', alpha=0.7)
plt.plot(test['GHI_Wm2'].values[:500], label=f'Modelo WRF original (rRMSE={rmse_test_original:.2f})', color='b', alpha=0.6)
plt.legend()
plt.xlabel('Número de muestra')
plt.ylabel('GHI [W/m²]')
plt.title('Comparación en conjunto Test  Buenos Aires (72h pronóstico)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()