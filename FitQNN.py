"""
qrnn_multi_horizon.py

QRNN multi-horizon (72h) para corrección de GHI.
- Lee tus archivos (modelados/ medidos/)
- Intenta construir dataset de secuencias por corrida (72 horas)
- Entrena una red que predice cuantiles (q10, q50, q90) para cada hora [1..72]
- Guarda scaler y modelo Keras

Requisitos:
pip install tensorflow pandas numpy matplotlib scikit-learn joblib openpyxl
"""

import os
import glob
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# TU MÓDULO DE MÉTRICAS (usa las funciones que ya tienes)
import Metrics as ms

# ============================
# Configuración / Parámetros
# ============================
SITE = 'BA'
N_H = 72                 # horizontes a predecir
QUANTILES = [0.1,0.5,0.9]
FEATURES = ['T2','TSLB']  # features por cada instante de inicialización (puedes agregar más)
# NOTA: asumimos que el forecast para cada lead (h=1..72) aparece en columna 'GHI_Wm2'
# en registros con different valid_time (long format). El script arma secuencias por run.
TEST_SIZE = 0.2
VAL_SIZE = 0.125  # respecto al conjunto de entrenamiento (luego usaremos split en train/val)
RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 300
MODEL_SAVE_DIR = 'models_qrnn'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ============================
# === 1. Cargar datos ===
# ============================
# Modelados (forecasts)
archivos = glob.glob(f"modelados/{SITE}*.csv")
if len(archivos) == 0:
    raise FileNotFoundError(f"No se encontraron archivos en modelados/{SITE}*.csv")

df_all = pd.concat([pd.read_csv(a) for a in archivos], ignore_index=True)
# Arreglar nombres si es necesario
# Intentamos convertir valid_time si existe
if 'valid_time' in df_all.columns:
    df_all['valid_time'] = pd.to_datetime(df_all['valid_time']) - timedelta(minutes=60)
else:
    # intentar otros nombres comunes
    for c in ['valid', 'validDate', 'valid_datetime', 'time']:
        if c in df_all.columns:
            df_all['valid_time'] = pd.to_datetime(df_all[c])
            break
    if 'valid_time' not in df_all.columns:
        raise KeyError("No se encontró columna 'valid_time' en df_all; adapta el script para tu formato.")

# Medidas
archivosMedidos = glob.glob(f"medidas/{SITE}*.xlsx")
if len(archivosMedidos) == 0:
    raise FileNotFoundError(f"No se encontraron archivos en medidas/{SITE}*.xlsx")

d = pd.concat([pd.read_excel(a) for a in archivosMedidos], ignore_index=True)
# Normalizar nombres
# Si las columnas son datetime, ghi, sza, GHIcc
# Si no, revisar manualmente
expected = ['datetime','ghi','sza','GHIcc']
if set(expected).issubset(set(d.columns)):
    d = d[expected]
else:
    # intentar adivinar
    cols = list(d.columns)
    # buscamos columna datetime
    dt_col = next((c for c in cols if 'date' in c.lower() or 'time' in c.lower()), None)
    ghi_col = next((c for c in cols if c.lower().startswith('ghi')), None)
    sza_col = next((c for c in cols if 'sza' in c.lower()), None)
    if dt_col is None or ghi_col is None:
        raise KeyError("No pude encontrar columnas de datetime/ghi en archivos medidos. Revisa nombres.")
    d = d[[dt_col, ghi_col, sza_col]] if sza_col else d[[dt_col, ghi_col]]
    d.columns = ['datetime','ghi','sza'] if sza_col else ['datetime','ghi','sza']
d['datetime'] = pd.to_datetime(d['datetime'])
# Filtrar SZA
if 'sza' in d.columns:
    d = d[d['sza'] < 90]

# ============================
# === A. Construir X_seq, Y_seq (heurísticas)
# ============================
# Objetivo: para cada corrida (t0), construir sample con features en t0 y como target las GHI reales en t0+1..t0+72
# Heurísticas (en orden):
# 1) Si existe columna 'init_time' o 'run_time' o 'forecast_reference_time' la usamos como agrupador y pivotamos a wide.
# 2) Si no existe, intentamos agrupar por bloques regulares de N_H filas por cada 'run' identificando jumps en valid_time grandes.
# 3) Como fallback, uso la fila como muestra única (modelo single-output). (Advertencia:esto reduce a single-horizon)

df_all_cols = df_all.columns.str.lower().tolist()

# 1) ¿Hay columna init/run?
group_col = None
for cand in ['init_time','run_time','reference_time','forecast_time','forecast_init','init']:
    if cand in df_all_cols:
        # use original-case column name
        group_col = [c for c in df_all.columns if c.lower()==cand][0]
        print(f"Usando columna de inicialización encontrada: {group_col}")
        break

# 2) ¿Hay columna 'lead' o 'forecast_hour'?
lead_col = None
for cand in ['lead','lead_time','forecast_hour','fhour','lead_h']:
    if cand in df_all_cols:
        lead_col = [c for c in df_all.columns if c.lower()==cand][0]
        print(f"Found lead column: {lead_col}")
        break

# Prepara df_all: requiere columna GHI_Wm2 (forecast). Debe existir pues tu pipeline la usaba.
if 'GHI_Wm2' not in df_all.columns and 'ghi_wm2' in df_all.columns.str.lower():
    # intenta encontrar y renombrar
    col = [c for c in df_all.columns if c.lower()=='ghi_wm2'][0]
    df_all.rename(columns={col:'GHI_Wm2'}, inplace=True)

if 'GHI_Wm2' not in df_all.columns:
    # Buscar otra columna con 'ghi' y 'wm2' o solo 'ghi'
    ghi_col = next((c for c in df_all.columns if 'ghi' in c.lower()), None)
    if ghi_col is None:
        raise KeyError("No se encontró columna de forecast GHI en df_all (busqué 'GHI_Wm2' o columnas con 'ghi').")
    else:
        print(f"Renombrando columna {ghi_col} -> GHI_Wm2")
        df_all.rename(columns={ghi_col:'GHI_Wm2'}, inplace=True)

# Merge forecasts with observations based on valid_time (inner join)
df_merged = pd.merge(df_all, d[['datetime','ghi']], left_on='valid_time', right_on='datetime', how='inner')
df_merged = df_merged.dropna(subset=['ghi'])

# Try method 1: if group_col and lead_col exist => pivot wide
X_seq = []
Y_seq = []
meta = []  # info for debugging: init_time etc.

if group_col is not None and lead_col is not None:
    print("Intentando pivotar por (group_col, lead_col) => formato wide")
    try:
        # Ensure proper datetime
        df_merged[group_col] = pd.to_datetime(df_merged[group_col])
        # Keep only necessary columns: group_col, lead_col, GHI_Wm2, valid_time, ghi (observed)
        tmp = df_merged[[group_col, lead_col, 'GHI_Wm2', 'valid_time', 'ghi'] + [c for c in FEATURES if c in df_merged.columns]]
        # pivot forecast to columns per lead
        pivot_fc = tmp.pivot_table(index=group_col, columns=lead_col, values='GHI_Wm2')
        # pivot observed ghi at valid_time - we want targets corresponding to lead 1..N_H
        # Build an ordered list of lead values for 1..N_H if leads numeric
        leads_sorted = sorted([c for c in pivot_fc.columns if not pd.isnull(c)])
        # select desired first N_H if possible
        if len(leads_sorted) < N_H:
            print(f"Advertencia: sólo encontré {len(leads_sorted)} leads por corrida (esperado {N_H}). Ajustando N_H a {len(leads_sorted)}.")
            N_H = len(leads_sorted)
        pivot_fc = pivot_fc.loc[:, leads_sorted[:N_H]]
        # Prepare features per run: take FEATURES from first row of each group (init)
        feats = tmp.groupby(group_col)[FEATURES].first()
        # For observed targets: necesitamos, por cada init, el ghi observado para los corresponding valid_times
        # Construimos mapping valid_time per (init,lead) by using tmp
        valid_map = tmp.pivot_table(index=group_col, columns=lead_col, values='valid_time')
        obs_map = tmp.pivot_table(index=group_col, columns=lead_col, values='ghi')
        # Filter only runs that have complete N_H observed values
        complete_runs = obs_map.dropna(subset=leads_sorted[:N_H]).index
        pivot_fc = pivot_fc.loc[complete_runs]
        feats = feats.loc[complete_runs]
        obs_map = obs_map.loc[complete_runs, leads_sorted[:N_H]]
        # Build arrays
        X_seq = feats.values  # shape (n_runs, n_features)
        Y_seq = obs_map.values  # shape (n_runs, N_H)
        meta = list(complete_runs)
        print(f"Construidos {len(X_seq)} muestras (runs) con {N_H} horizontes.")
    except Exception as e:
        print("Falló pivot por group_col+lead_col:", e)
        X_seq = []
        Y_seq = []

# If previous failed, try method 2: detect run boundaries using gaps in valid_time (works if data is long)
if len(X_seq)==0:
    print("Intentando heurística por gaps en valid_time (formato long sin init column).")
    # Ordenar por valid_time
    df_sorted = df_merged.sort_values('valid_time').reset_index(drop=True)
    # Se considera inicio de nueva corrida si la diferencia entre filas consecutivas es mayor a umbral (ej 10 horas)
    dt = df_sorted['valid_time'].diff().dt.total_seconds().fillna(0) / 3600.0
    # Umbral: si hay saltos > 8 horas consideramos nueva corrida (porque runs son 6h o 12h)
    threshold_hours = 8.0
    new_run_idx = (dt > threshold_hours).cumsum()
    df_sorted['run_id'] = new_run_idx
    runs = df_sorted.groupby('run_id')
    samples = 0
    X_list = []
    Y_list = []
    meta = []
    for run_id, group in runs:
        group = group.sort_values('valid_time')
        # sólo consideramos runs con al menos N_H filas consecutivas
        if len(group) >= N_H:
            # tomamos la primera subsecuencia completa de longitud N_H
            seq = group.iloc[:N_H]
            # features from initial time
            if not set(FEATURES).issubset(seq.columns):
                print("Warning: no todas las FEATURES están en df; usa las que existan.")
                feat_cols = [c for c in FEATURES if c in seq.columns]
            else:
                feat_cols = FEATURES
            X_list.append(seq[feat_cols].iloc[0].values)
            Y_list.append(seq['ghi'].values[:N_H])
            meta.append(seq['valid_time'].iloc[0])  # init time
            samples += 1
    if samples>0:
        X_seq = np.vstack(X_list)
        Y_seq = np.vstack(Y_list)
        print(f"Construidos {X_seq.shape[0]} muestras por heurística de gaps (cada una con {N_H} horizontes).")
    else:
        print("No se pudieron construir runs completos con la heurística de gaps.")

# Fallback: si todavía vacío, reducir a single-output (advertencia)
if len(X_seq)==0:
    print("No se pudo construir dataset multi-horizon. Se crea dataset single-output (fila a fila) como fallback.")
    # Usaremos FEATURES y target ghi observado en ese valid_time -> modelo QRNN single-horizon
    # Convertir a arrays
    feats = [c for c in FEATURES if c in df_merged.columns]
    X_seq = df_merged[feats].values
    Y_seq = df_merged['ghi'].values.reshape(-1,1)
    N_H = 1
    QUANTILES = [0.1,0.5,0.9]
    print(f"Numero de muestras: {len(X_seq)}. N_H={N_H} (single-output).")

# ============================
# === B. Preparar datos para ML (escalado y splits)
# ============================
# X_seq shape: (n_samples, n_features)
# Y_seq shape: (n_samples, N_H)

X = X_seq
Y = Y_seq

# Check shapes
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Split train/test/val
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Scaling: escalar features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Guardar scaler
joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, f'scaler_{SITE}.pkl'))
print("Scaler guardado.")

# ============================
# === C. Modelo QRNN multi-horizon (Keras)
# ============================
def quantile_loss(q):
    # return a loss function for quantile q
    def loss(y_true, y_pred):
        err = y_true - y_pred
        # pinball
        return tf.reduce_mean(tf.maximum(q * err, (q - 1) * err))
    return loss

def build_qrnn_model(input_dim, n_horizons, quantiles, hidden_layers=[128,64], dropout=0.1):
    inputs = keras.Input(shape=(input_dim,), name='inputs')
    x = inputs
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
    # outputs: one Dense(n_horizons) per quantile
    outputs = [layers.Dense(n_horizons, name=f'q{int(q*100)}')(x) for q in quantiles]
    model = keras.Model(inputs=inputs, outputs=outputs)
    # compile with multiple losses
    losses = {f'q{int(q*100)}': quantile_loss(q) for q in quantiles}
    # Keras expects either list of losses (in same order) or dict keyed by output names
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=[quantile_loss(q) for q in quantiles])
    return model

input_dim = X_train_s.shape[1]
model = build_qrnn_model(input_dim=input_dim, n_horizons=N_H, quantiles=QUANTILES, hidden_layers=[128,64], dropout=0.1)
model.summary()

# Prepare y for training: list of arrays (one per quantile), each with shape (n_samples, n_horizons)
y_train_list = [Y_train for _ in QUANTILES]
y_val_list = [Y_val for _ in QUANTILES]

# Callbacks
early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# ============================
# === D. Entrenamiento
# ============================
history = model.fit(
    X_train_s,
    y_train_list,
    validation_data=(X_val_s, y_val_list),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early, reduce_lr],
    verbose=2
)

# Guardar modelo
model.save(os.path.join(MODEL_SAVE_DIR, f'qrnn_multi_h_{SITE}.keras'))
print("Modelo QRNN guardado.")

# ============================
# === E. Predicción y evaluación
# ============================
# Predecir sobre test
y_pred_list = model.predict(X_test_s)
# y_pred_list is list with len=number of quantiles; each element shape (n_samples, N_H)
# Rearrange: median is index of quantile 0.5
q_to_idx = {q:i for i,q in enumerate(QUANTILES)}
median_idx = q_to_idx.get(0.5, None)
if median_idx is None:
    # If no 0.5 in QUANTILES, choose middle quantile
    median_idx = len(QUANTILES)//2
y_pred_median = y_pred_list[median_idx]  # shape (n_samples, N_H)

# Evaluate rrmsd and rmbe for median flattened (compare to Y_test)
# Si N_H > 1, calculamos metricas por horizonte y promediadas
def evaluate_multi(y_true, y_pred, ms_module):
    # y_true, y_pred: (n_samples, N_H)
    n_h = y_true.shape[1]
    rrmsd_per_h = []
    rmbe_per_h = []
    for h in range(n_h):
        rr = ms_module.rrmsd(y_true[:,h], y_pred[:,h])
        rmbe = ms_module.rmbe(y_true[:,h], y_pred[:,h])
        rrmsd_per_h.append(rr)
        rmbe_per_h.append(rmbe)
    return np.array(rrmsd_per_h), np.array(rmbe_per_h)

rrmsd_h, rmbe_h = evaluate_multi(Y_test, y_pred_median, ms)
print("RRMSD por horizonte (primeras 10):", rrmsd_h[:10])
print("RRBE por horizonte (primeras 10):", rmbe_h[:10])
print(f"RRMSD promedio (mediana): {np.nanmean(rrmsd_h):.4f}, RBE promedio: {np.nanmean(rmbe_h):.4f}")

# Comparación con modelo original (GHI_Wm2) si aplicable:
# Si en fallback single-output, Y_test shape (n,1), test original forecast must be available:
if N_H==1:
    # intentar comparar con forecast original (si tenías esa columna)
    print("Comparación contra forecast original: (single-output case)")
    # No tenemos el forecast original en X_test; si lo deseas, guarda la columna antes del pivot.
else:
    print("Comparación con modelo original por horizonte no implementada (requiere matriz forecast original por run).")

# Evaluación de cobertura del intervalo q10-q90 (si están presentes)
if 0.1 in QUANTILES and 0.9 in QUANTILES:
    idx10 = q_to_idx[0.1]
    idx90 = q_to_idx[0.9]
    y_q10 = y_pred_list[idx10]
    y_q90 = y_pred_list[idx90]
    # coverage: fracción de observaciones que caen dentro del intervalo [q10,q90]
    coverage_per_h = np.mean((Y_test >= y_q10) & (Y_test <= y_q90), axis=0)
    mean_coverage = np.mean(coverage_per_h)
    mean_width = np.mean(y_q90 - y_q10)
    print(f"Cobertura media del intervalo 10%-90%: {mean_coverage*100:.2f}% (ancho medio: {mean_width:.2f} W/m2)")
else:
    print("No se proporcionaron quantiles 0.1/0.9 — no se calcula cobertura.")

# ============================
# === F. Visualizaciones
# ============================
# Plot loss
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Época')
plt.ylabel('Loss (suma pinball sobre quantiles)')
plt.legend()
plt.title('Historia de entrenamiento')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot sample of test: para las primeras N_plot muestras, mostrar observados, mediana, intervalos
N_plot = 300  # puntos a mostrar
if N_H == 1:
    # plot en secuencia de muestras
    plt.figure(figsize=(12,4))
    plt.plot(Y_test.flatten()[:N_plot], label='Observado', color='k')
    plt.plot(y_pred_median.flatten()[:N_plot], label='QRNN mediana', color='r')
    if 0.1 in QUANTILES and 0.9 in QUANTILES:
        plt.fill_between(np.arange(N_plot),
                         y_pred_list[q_to_idx[0.1]].flatten()[:N_plot],
                         y_pred_list[q_to_idx[0.9]].flatten()[:N_plot], alpha=0.3, label='[q10,q90]')
    plt.legend()
    plt.title('QRNN - Test (single-output)')
    plt.xlabel('Muestra')
    plt.ylabel('GHI [W/m2]')
    plt.tight_layout()
    plt.show()
else:
    # seleccionar una corrida de test y graficar horizontes (por ejemplo primer sample)
    sample_idx = 0
    # mostrar las N_H horas de la sample
    horizons = np.arange(1, N_H+1)
    plt.figure(figsize=(12,4))
    plt.plot(horizons, Y_test[sample_idx,:], marker='o', label='Observado (horiz.)', color='k')
    plt.plot(horizons, y_pred_median[sample_idx,:], marker='o', label='Mediana', color='r')
    if 0.1 in QUANTILES and 0.9 in QUANTILES:
        plt.fill_between(horizons,
                         y_pred_list[q_to_idx[0.1]][sample_idx,:],
                         y_pred_list[q_to_idx[0.9]][sample_idx,:], alpha=0.2, label='[q10,q90]')
    plt.xlabel('Lead (h)')
    plt.ylabel('GHI [W/m2]')
    plt.legend()
    plt.grid(True)
    plt.title(f'Sample test #{sample_idx} — secuencia {N_H}h')
    plt.tight_layout()
    plt.show()

# ============================
# === G. Guardar predicciones ejemplo (CSV)
# ============================
# Guardar predicciones de test (mediana y quantiles) en csv para análisis posterior
out_df = pd.DataFrame()
n_test = X_test.shape[0]
for i,q in enumerate(QUANTILES):
    preds = y_pred_list[i]  # shape (n_test, N_H)
    # crear columnas q{q}_h{h}
    for h in range(N_H):
        out_df[f"q{int(q*100)}_h{h+1}"] = preds[:,h]
# agregar observados (flatten por horizonte): guardar como columnas y si Y_test tiene shape (n_test,N_H)
for h in range(N_H):
    out_df[f"obs_h{h+1}"] = Y_test[:,h]
# Guardar
out_csv = os.path.join(MODEL_SAVE_DIR, f'preds_test_{SITE}.csv')
out_df.to_csv(out_csv, index=False)
print(f"Predicciones de test guardadas en {out_csv}")

print("Proceso finalizado.")

