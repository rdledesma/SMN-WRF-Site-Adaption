import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import joblib
import Metrics as ms   # tus métricas personalizadas

# === 1. Cargar datos ===
df_all = pd.DataFrame()
for x in ['01','02','03','04','05','06','07','12']:
    mod = pd.read_csv(f'modelados/BA_2023_{x}_all.csv')
    mod['month'] = x
    df_all = pd.concat([df_all, mod])

df_all['valid_time'] = pd.to_datetime(df_all.valid_time) - timedelta(minutes=60)

d = pd.read_excel('medidas/BA_2023_QC_60min.xlsx')
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
features = ['T2','PSFC', 'ACLWDNB', 'ACLWUPB', 'ACSWDNB', 'TSLB', 'SMOIS','GHI_Wm2']
target = 'ghi'

train = df[df['month'].isin(['01','02','03','04','05','06'])]
val   = df[df['month'] == '07']
test  = df[df['month'] == '12']

X_train, y_train = train[features], train[target]
X_val, y_val     = val[features], val[target]
X_test, y_test   = test[features], test[target]

# === 3. Escalado ===
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# === 4. Definimos grilla de hiperparámetros ===
param_grid = {
    'hidden_layer_sizes': [(5,10), (10,15), (15,20)],
    'activation': ['relu'],
    'learning_rate_init': [ 0.01],
    'max_iter': [200]
}

# === 5. Entrenar y evaluar manualmente ===
mejor_modelo = None
mejor_params = None
mejor_rmse = np.inf
mejor_rmbe = None

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
    if rmse < mejor_rmse:
        mejor_rmse = rmse
        mejor_rmbe = rmbe
        mejor_modelo = mlp
        mejor_params = params

print("\n=== Mejor modelo según rrmsd ===")
print(mejor_params)
print(f"rrmsd={mejor_rmse:.3f}, rmbe={mejor_rmbe:.3f}")

# === 6. Guardar scaler y modelo ===
joblib.dump(scaler, 'scaler_MLP_BA2023.pkl')
joblib.dump(mejor_modelo, 'MLP_BA2023_best.pkl')
print("Modelos guardados: scaler_MLP_BA2023.pkl, MLP_BA2023_best.pkl")




# === 7. Evaluación final en test ===
y_pred_test = mejor_modelo.predict(X_test_s)
rmse_test = ms.rrmsd(y_test, y_pred_test)
rmse_test_original = ms.rrmsd(y_test, test.GHI_Wm2.values)

rmbe_test = ms.rmbe(y_test, y_pred_test)
rmbe_test_original = ms.rmbe(y_test, test.GHI_Wm2.values)
print(f"Test - rrmsd={rmse_test:.3f}, rmbe={rmbe_test:.3f}")
print(f"TestOriginal - rrmsd={rmse_test_original:.3f}, rmbe={rmbe_test:.3f}")

# === 8. Graficar resultados del test ===
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:500], label='Observado', color='k')
plt.plot(y_pred_test[:500], label=f'Pronosticado + MLP rRMSE {rmse_test:.2f}', color='r', alpha=0.7)
plt.plot(test.GHI_Wm2.values[:500], label=f'Pronosticado rRMSE {rmse_test_original:.2f}', color='b', alpha=0.6)
plt.legend()
plt.xlabel('Número de muestra')
plt.title('Comparación en conjunto Test para Buenos Aires 72 hs de una corrida)')
plt.show()
