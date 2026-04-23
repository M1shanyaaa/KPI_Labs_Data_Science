"""
Лабораторна робота №6 — IV рівень складності (20 балів)
R&D: Оптимізація архітектури штучної нейронної мережі
Показники: точність (MSE/MAE) та час навчання
Дані: РЕАЛЬНИЙ часовий ряд — історичні ціни акцій Apple (AAPL.Close)
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import urllib.request
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ─── 1. ЗАВАНТАЖЕННЯ РЕАЛЬНОГО ЧАСОВОГО РЯДУ ───────────────────────────────
print("Завантаження реальних фінансових даних...")
# Використовуємо публічний датасет з цінами акцій Apple
url = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"

try:
    df = pd.read_csv(url)
    # Беремо колонку з ціною закриття (Close price)
    series = df['AAPL.Close'].values.copy()
    print(f"Дані успішно завантажено. Кількість записів: {len(series)}")
except Exception as e:
    print(f"Помилка завантаження даних: {e}")
    print("Перевірте підключення до інтернету.")
    exit(1)

# Додаємо трохи штучних аномалій (викидів), щоб ускладнити завдання для мережі,
# імітуючи різкі стрибки на ринку (близько 5% даних)
np.random.seed(42)
N = len(series)
anomaly_idx = np.random.choice(N, int(0.05 * N), replace=False)
series[anomaly_idx] += np.random.uniform(-15, 15, len(anomaly_idx))

# ─── 2. ПІДГОТОВКА ДАНИХ ───────────────────────────────────────────────────
WINDOW = 20       # розмір вікна (вхідних кроків)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

X, y = [], []
for i in range(WINDOW, len(scaled)):
    X.append(scaled[i - WINDOW:i])
    y.append(scaled[i])
X, y = np.array(X), np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ─── 3. АРХІТЕКТУРИ ДЛЯ ДОСЛІДЖЕННЯ ──────────────────────────────────────
architectures = {
    "Tiny  [32]":          [32],
    "Small [64,32]":       [64, 32],
    "Medium [128,64,32]":  [128, 64, 32],
    "Large [256,128,64]":  [256, 128, 64],
    "Deep  [128,64,32,16]":[128, 64, 32, 16],
}

# ─── 4. НАВЧАННЯ ТА ОЦІНКА ────────────────────────────────────────────────
results = {}
es = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)

print(f"\n{'Архітектура':<26} {'MSE':>8} {'MAE':>8} {'Час(с)':>8} {'Епох':>6}")
print("─" * 60)

for name, layers in architectures.items():
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_shape=(WINDOW,)))
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    t0 = time.time()
    hist = model.fit(X_train, y_train, epochs=200, batch_size=64,
                     validation_split=0.1, callbacks=[es], verbose=0)
    elapsed = time.time() - t0
    epochs_done = len(hist.history['loss'])

    pred_scaled = model.predict(X_test, verbose=0).flatten()
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = mean_squared_error(real, pred)
    mae = mean_absolute_error(real, pred)
    results[name] = {'mse': mse, 'mae': mae, 'time': elapsed,
                     'epochs': epochs_done, 'pred': pred,
                     'loss': hist.history['loss'],
                     'val_loss': hist.history.get('val_loss', [])}
    print(f"{name:<26} {mse:>8.3f} {mae:>8.3f} {elapsed:>8.2f} {epochs_done:>6}")

# ─── 5. ВІЗУАЛІЗАЦІЯ ──────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("R&D: Оптимізація архітектури ANN — Аналіз ефективності (Акції AAPL)", fontsize=14)

real_vals = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

# --- 5.1 Прогнозування: кращої моделі ---
best_name = min(results, key=lambda k: results[k]['mse'])
ax = axes[0, 0]
ax.plot(real_vals[:300], label='Реальний ряд (Apple Close Price)', color='black', lw=1.5)
ax.plot(results[best_name]['pred'][:300], label=f'Прогноз ({best_name})', color='red', lw=1.5, ls='--')
ax.set_title(f"Прогнозування — найкраща архітектура: {best_name}")
ax.set_xlabel("Кроки часу"); ax.set_ylabel("Ціна ($)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# --- 5.2 MSE по архітектурах ---
ax = axes[0, 1]
names = list(results.keys())
mse_vals = [results[n]['mse'] for n in names]
bars = ax.bar(range(len(names)), mse_vals, color=colors)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.split('[')[0].strip() for n in names], rotation=15, fontsize=8)
ax.set_title("MSE залежно від архітектури"); ax.set_ylabel("MSE"); ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, mse_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{v:.2f}', ha='center', fontsize=7)

# --- 5.3 Час навчання ---
ax = axes[1, 0]
time_vals = [results[n]['time'] for n in names]
bars = ax.bar(range(len(names)), time_vals, color=colors)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.split('[')[0].strip() for n in names], rotation=15, fontsize=8)
ax.set_title("Час навчання залежно від архітектури"); ax.set_ylabel("Час (с)"); ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, time_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{v:.1f}', ha='center', fontsize=7)

# --- 5.4 Компроміс MSE/Час (Pareto) ---
ax = axes[1, 1]
for i, (name, c) in enumerate(zip(names, colors)):
    ax.scatter(results[name]['time'], results[name]['mse'], color=c, s=120, zorder=5, label=name.split('[')[0].strip())
    ax.annotate(name.split('[')[0].strip(), (results[name]['time'], results[name]['mse']),
                textcoords="offset points", xytext=(5, 5), fontsize=7)
ax.set_xlabel("Час навчання (с)"); ax.set_ylabel("MSE")
ax.set_title("Pareto: MSE vs Час (компроміс точність/швидкість)"); ax.grid(True, alpha=0.3)

# --- 5.5 Криві навчання (loss) ---
ax = axes[2, 0]
for name, c in zip(names, colors):
    ax.plot(results[name]['loss'], color=c, label=name.split('[')[0].strip(), lw=1.5)
ax.set_title("Криві навчання (Train Loss)"); ax.set_xlabel("Епохи"); ax.set_ylabel("MSE Loss")
ax.legend(fontsize=7); ax.set_yscale('log'); ax.grid(True, alpha=0.3)

# --- 5.6 MAE порівняння ---
ax = axes[2, 1]
mae_vals = [results[n]['mae'] for n in names]
epochs_vals = [results[n]['epochs'] for n in names]
ax2 = ax.twinx()
ax.bar(range(len(names)), mae_vals, color=colors, alpha=0.7, label='MAE')
ax2.plot(range(len(names)), epochs_vals, 'ko--', lw=2, label='Кількість епох')
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.split('[')[0].strip() for n in names], rotation=15, fontsize=8)
ax.set_ylabel("MAE"); ax2.set_ylabel("Кількість епох")
ax.set_title("MAE та кількість епох навчання"); ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ann_optimization_results.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nГрафік збережено: {save_path}")

# ─── 6. ВИСНОВКИ ──────────────────────────────────────────────────────────
best_mse = min(results, key=lambda k: results[k]['mse'])
best_time = min(results, key=lambda k: results[k]['time'])
worst_mse = max(results, key=lambda k: results[k]['mse'])

print("\n" + "═" * 60)
print("ВИСНОВКИ R&D ДОСЛІДЖЕННЯ")
print("═" * 60)
print(f"  Найточніша архітектура : {best_mse} (MSE={results[best_mse]['mse']:.3f})")
print(f"  Найшвидша архітектура  : {best_time} (час={results[best_time]['time']:.2f}с)")
print(f"  Найгірша точність      : {worst_mse} (MSE={results[worst_mse]['mse']:.3f})")
mse_gain = (results[worst_mse]['mse'] - results[best_mse]['mse']) / results[worst_mse]['mse'] * 100
time_cost = results[best_mse]['time'] / results[best_time]['time']
print(f"  Приріст точності (MSE) : {mse_gain:.1f}%")
print(f"  Часова ціна кращої моделі: x{time_cost:.1f} відносно найшвидшої")
print(f"\n  Оптимальний вибір (Pareto): архітектура з балансом MSE/Час —")
# Знаходимо pareto-optimal: нормалізуємо і беремо мін суму
norm_mse = np.array([results[n]['mse'] for n in names])
norm_time = np.array([results[n]['time'] for n in names])
norm_mse = (norm_mse - norm_mse.min()) / (norm_mse.max() - norm_mse.min() + 1e-9)
norm_time = (norm_time - norm_time.min()) / (norm_time.max() - norm_time.min() + 1e-9)
pareto_score = norm_mse + norm_time
best_pareto = names[np.argmin(pareto_score)]
print(f"  → {best_pareto}")
print("═" * 60)