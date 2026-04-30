"""
Лабораторна робота №7 — II рівень складності (9 балів)
Data_Set_11.xlsx — аналіз часових рядів продажів торгової компанії
Методи: МНК (поліноміальна апроксимація), ARIMA, ANN (MLP)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# ─── 1. Парсинг та попередній аналіз ────────────────────────────────────────
df = pd.read_excel("Data_Set_11.xlsx", sheet_name="Orders", parse_dates=["Order Date"])
df.columns = df.columns.str.strip()
df.dropna(subset=["Sales", "Profit"], inplace=True)

print("=" * 60)
print("ПОПЕРЕДНІЙ АНАЛІЗ ДАНИХ")
print("=" * 60)
print(f"Записів: {len(df)}, Регіонів: {df['Region'].nunique()}")
print(f"Діапазон дат: {df['Order Date'].min().date()} → {df['Order Date'].max().date()}")
print(f"\nПоказники ефективності:")
print(df[["Sales", "Profit", "Discount", "Order Quantity"]].describe().round(2))

# ─── 2. Агрегація по місяцях ────────────────────────────────────────────────
df["YearMonth"] = df["Order Date"].dt.to_period("M")
monthly = df.groupby("YearMonth")[["Sales", "Profit"]].sum().reset_index()
monthly["t"] = np.arange(len(monthly))
monthly["date"] = monthly["YearMonth"].dt.to_timestamp()

# ─── 3. Аналіз по регіонах ───────────────────────
region_summary = (
    df.groupby("Region")[["Sales", "Profit"]]
    .agg(["sum", "mean"])
    .round(2)
)
region_summary.columns = ["Sales_Sum", "Sales_Mean", "Profit_Sum", "Profit_Mean"]
print("\nПоказники за регіонами:")
print(region_summary.to_string())

# ─── 4. Тест стаціонарності ───────────────────────────────────
adf_stat, adf_p, *_ = adfuller(monthly["Sales"])
print(f"\nADF-тест (Sales): stat={adf_stat:.3f}, p={adf_p:.4f} → "
      f"{'стаціонарний' if adf_p < 0.05 else 'нестаціонарний'}")

# ─── 5. МНК — поліноміальна апроксимація ───────────────────────────
t = monthly["t"].values
y = monthly["Sales"].values
deg = 3
coeffs = np.polyfit(t, y, deg)
poly = np.poly1d(coeffs)
y_mnk = poly(t)

t_future = np.arange(len(t), len(t) + 6)
y_future_mnk = poly(t_future)

rmse_mnk = np.sqrt(mean_squared_error(y, y_mnk))
mae_mnk  = mean_absolute_error(y, y_mnk)
print(f"\nМНК (поліном {deg}-го ступеня): RMSE={rmse_mnk:.1f}, MAE={mae_mnk:.1f}")
print(f"Коефіцієнти: {np.round(coeffs, 4)}")

# ─── 6. ARIMA ────────────────────────────────────────────────────────────────
arima_model = ARIMA(y, order=(2, 1, 2)).fit()
y_arima = arima_model.fittedvalues
forecast_arima = arima_model.forecast(steps=6)

rmse_arima = np.sqrt(mean_squared_error(y[1:], y_arima[1:]))
mae_arima  = mean_absolute_error(y[1:], y_arima[1:])
print(f"\nARIMA(2,1,2): RMSE={rmse_arima:.1f}, MAE={mae_arima:.1f}")
print(f"Прогноз +6 міс: {np.round(np.array(forecast_arima), 0)}")

# ─── 7. ANN (MLP) ────────────────────────────────────────────────────────────
WINDOW = 6
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

X_ann = np.array([y_scaled[i:i+WINDOW] for i in range(len(y_scaled)-WINDOW)])
Y_ann = y_scaled[WINDOW:]

split = int(len(X_ann) * 0.8)
X_tr, X_te = X_ann[:split], X_ann[split:]
Y_tr, Y_te = Y_ann[:split], Y_ann[split:]

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42,
                   learning_rate_init=0.001, early_stopping=True, n_iter_no_change=30)
mlp.fit(X_tr, Y_tr)

y_ann_pred_scaled = mlp.predict(X_ann)
y_ann_pred = scaler.inverse_transform(y_ann_pred_scaled.reshape(-1,1)).flatten()
y_ann_true = y[WINDOW:]

# Forecast 6 steps ahead
last_window = y_scaled[-WINDOW:].copy()
ann_forecast = []
for _ in range(6):
    pred = mlp.predict(last_window.reshape(1, -1))[0]
    ann_forecast.append(pred)
    last_window = np.roll(last_window, -1)
    last_window[-1] = pred
ann_forecast = scaler.inverse_transform(np.array(ann_forecast).reshape(-1,1)).flatten()

rmse_ann = np.sqrt(mean_squared_error(y_ann_true, y_ann_pred))
mae_ann  = mean_absolute_error(y_ann_true, y_ann_pred)
print(f"\nANN (MLP 64-32): RMSE={rmse_ann:.1f}, MAE={mae_ann:.1f}")
print(f"Прогноз +6 міс: {np.round(ann_forecast, 0)}")

# ─── 8. Порівняльна таблиця ──────────────────────────────────────────────────
future_dates = pd.period_range(monthly["YearMonth"].max() + 1, periods=6, freq="M")
comparison = pd.DataFrame({
    "Місяць":       [str(d) for d in future_dates],
    "МНК":          np.round(y_future_mnk, 0),
    "ARIMA":        np.round(np.array(forecast_arima), 0),
    "ANN":          np.round(ann_forecast, 0),
})
print("\nПРОГНОЗ ПРОДАЖІВ НА 6 МІСЯЦІВ:")
print(comparison.to_string(index=False))

metrics = pd.DataFrame({
    "Метод": ["МНК", "ARIMA", "ANN"],
    "RMSE":  [round(rmse_mnk,1), round(rmse_arima,1), round(rmse_ann,1)],
    "MAE":   [round(mae_mnk,1),  round(mae_arima,1),  round(mae_ann,1)],
})
print("\nПОРІВНЯЛЬНИЙ АНАЛІЗ МЕТОДІВ:")
print(metrics.to_string(index=False))
best = metrics.loc[metrics["RMSE"].idxmin(), "Метод"]
print(f"Найкращий метод за RMSE: {best}")

# ─── 9. Візуалізація ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Аналіз часового ряду продажів (Data_Set_11)", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

dates = monthly["date"].values
fut_ts = pd.date_range(monthly["date"].iloc[-1] + pd.DateOffset(months=1), periods=6, freq="ME")

# 9.1 МНК
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(dates, y, "b-o", ms=3, label="Факт")
ax1.plot(dates, y_mnk, "r--", lw=2, label=f"МНК (ступ. {deg})")
ax1.plot(fut_ts, y_future_mnk, "r:", lw=2, label="Прогноз МНК")
ax1.set_title(f"МНК | RMSE={rmse_mnk:.0f}")
ax1.legend(fontsize=8); ax1.tick_params(labelsize=7)

# 9.2 ARIMA
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(dates, y, "b-o", ms=3, label="Факт")
ax2.plot(dates[1:], y_arima[1:], "g--", lw=2, label="ARIMA fit")
ax2.plot(fut_ts, np.array(forecast_arima), "g:", lw=2, label="Прогноз ARIMA")
ax2.set_title(f"ARIMA(2,1,2) | RMSE={rmse_arima:.0f}")
ax2.legend(fontsize=8); ax2.tick_params(labelsize=7)

# 9.3 ANN
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(dates, y, "b-o", ms=3, label="Факт")
ax3.plot(dates[WINDOW:], y_ann_pred, "m--", lw=2, label="ANN fit")
ax3.plot(fut_ts, ann_forecast, "m:", lw=2, label="Прогноз ANN")
ax3.set_title(f"ANN (MLP) | RMSE={rmse_ann:.0f}")
ax3.legend(fontsize=8); ax3.tick_params(labelsize=7)

# 9.4 Порівняння прогнозів
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(dates[-12:], y[-12:], "b-o", ms=4, label="Факт (останні 12 міс)")
ax4.plot(fut_ts, y_future_mnk, "r-s", ms=5, label="МНК")
ax4.plot(fut_ts, np.array(forecast_arima), "g-^", ms=5, label="ARIMA")
ax4.plot(fut_ts, ann_forecast, "m-D", ms=5, label="ANN")
ax4.axvline(dates[-1], color="k", linestyle="--", lw=1)
ax4.set_title("Порівняння прогнозів (+6 міс)")
ax4.legend(fontsize=8); ax4.tick_params(labelsize=7)

# 9.5 Динаміка по регіонах
ax5 = fig.add_subplot(gs[2, 0])
region_monthly = df.groupby(["YearMonth", "Region"])["Sales"].sum().unstack(fill_value=0)
for reg in region_monthly.columns:
    ax5.plot(region_monthly.index.to_timestamp(), region_monthly[reg], label=reg, lw=1.5)
ax5.set_title("Динаміка продажів за регіонами")
ax5.legend(fontsize=7); ax5.tick_params(labelsize=7)

# 9.6 Метрики якості
ax6 = fig.add_subplot(gs[2, 1])
x_pos = np.arange(len(metrics))
bars = ax6.bar(x_pos - 0.2, metrics["RMSE"], 0.35, label="RMSE", color=["#e74c3c","#2ecc71","#9b59b6"])
ax6.bar(x_pos + 0.2, metrics["MAE"], 0.35, label="MAE", color=["#c0392b","#27ae60","#8e44ad"], alpha=0.7)
ax6.set_xticks(x_pos); ax6.set_xticklabels(metrics["Метод"])
ax6.set_title("Порівняльні метрики")
ax6.legend(fontsize=8); ax6.tick_params(labelsize=8)
for bar in bars:
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
             f"{bar.get_height():.0f}", ha="center", fontsize=7)

plt.savefig("lab7_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nГрафік збережено: lab7_results.png")