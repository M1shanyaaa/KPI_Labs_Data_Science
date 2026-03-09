import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
from datetime import datetime
import time
import matplotlib

matplotlib.use('TkAgg')
FILENAME = 'nbu_pln_history.csv'


def parse_pln_history():
    all_data = []

    # Тільки минулі та поточні місяці
    years_months = [
        "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
        "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
        "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
        "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
        "2026-01", "2026-02", "2026-03",
    ]

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/124.0.0.0 Safari/537.36'
        ),
        'Accept-Language': 'uk-UA,uk;q=0.9,en;q=0.8',
        'Accept': 'text/html,application/xhtml+xml',
        'Referer': 'https://index.minfin.com.ua/',
    }

    for ym in years_months:
        url = f"https://index.minfin.com.ua/ua/exchange/nbu/curr/pln/{ym}/"
        print(f"Парсинг даних за {ym}...")
        try:
            response = requests.get(url, headers=headers, timeout=20)

            # Пропускаємо майбутні місяці без помилки
            if response.status_code == 404:
                print(f" {ym} — ще не існує, пропускаємо")
                continue
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            table = soup.find('table', class_='zebra')
            if not table:
                table = soup.find('table')
            if not table:
                print(f"Таблицю не знайдено для {ym}")
                continue

            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    date_element = cols[0].find(string=True)
                    if date_element:
                        date_raw = date_element.strip()
                        price_raw = (
                            cols[1].get_text(strip=True)
                            .replace(',', '.')
                            .replace(' ', '')
                            .replace('\xa0', '')
                        )
                        try:
                            all_data.append({
                                'Date': datetime.strptime(date_raw, '%d.%m.%Y'),
                                'Price': float(price_raw)
                            })
                        except ValueError:
                            continue

            time.sleep(0.4)

        except Exception as e:
            print(f" Помилка при запиті {ym}: {e}")

    if not all_data:
        raise RuntimeError(
            " Дані не отримано.\n"
            "Спробуй відкрити в браузері:\n"
            "  https://index.minfin.com.ua/ua/exchange/nbu/curr/pln/2024-01/\n"
            "Якщо сторінка є — скинь HTML і я виправлю парсер."
        )

    df = pd.DataFrame(all_data).sort_values('Date').reset_index(drop=True)
    df.to_csv(FILENAME, index=False)
    print(f"\n✅ Зібрано {len(df)} записів, збережено у {FILENAME}")
    return df


# ЗАВАНТАЖЕННЯ ДАНИХ
if os.path.exists(FILENAME):
    print(f"Завантажую збережені дані з {FILENAME}")
    df = pd.read_csv(FILENAME)
    df['Date'] = pd.to_datetime(df['Date'])
else:
    print(f"Файл {FILENAME} не знайдено. Запускаю парсинг...")
    df = parse_pln_history()

print(f"Записів: {len(df)}, курс PLN: {df['Price'].min():.4f} – {df['Price'].max():.4f} UAH")


# R&D 3.1: Sliding MAD — виявлення аномалій
def inject_anomalies(series, rate=0.04, severity=4.0):
    data = series.copy()
    n_anomalies = max(1, int(len(data) * rate))
    std_dev = np.std(data)
    indices = np.random.choice(range(5, len(data) - 5), n_anomalies, replace=False)
    for idx in indices:
        direction = np.random.choice([1, -1])
        data[idx] += direction * (severity * std_dev + np.random.uniform(0.5, 1.5) * std_dev)
    return data, indices


def anomaly_detector(series, window=10, k=3.0):
    rolling_med = (series.rolling(window=window, center=True).median().ffill().bfill())

    def get_mad(x):
        m = np.median(x)
        return np.median(np.abs(x - m))

    rolling_mad = (series.rolling(window=window, center=True).apply(get_mad).ffill().bfill())
    anomalies = np.where(np.abs(series - rolling_med) > (k * rolling_mad * 1.4826))[0]
    cleaned = series.copy()
    for idx in anomalies:
        cleaned.iloc[idx] = rolling_med.iloc[idx]

    return anomalies, cleaned


np.random.seed(42)
df['Price_Anom'], true_idx = inject_anomalies(df['Price'].values)
det_idx, df['Price_Clean'] = anomaly_detector(pd.Series(df['Price_Anom']))

tp = len(set(det_idx) & set(true_idx))
fp = len(set(det_idx) - set(true_idx))
fn = len(set(true_idx) - set(det_idx))
pr = tp / (tp + fp + 1e-10)
rc = tp / (tp + fn + 1e-10)
f1 = 2 * pr * rc / (pr + rc + 1e-10)
print(f"\nSliding MAD → TP={tp}, FP={fp}, FN={fn}")
print(f"Precision={pr:.2f}, Recall={rc:.2f}, F1={f1:.2f}")

# Група 1.4–1.6: МНК — підбір ступеня полінома за Adj R²

n = len(df)
t = np.arange(n, dtype=float)
y_clean = df['Price_Clean'].values

best_deg, best_adj_r2, poly_results = 1, -np.inf, {}
for deg in range(1, 6):
    c = np.polyfit(t, y_clean, deg)
    p = np.poly1d(c)
    r2 = r2_score(y_clean, p(t))
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - deg - 1)
    poly_results[deg] = {
        "poly": p, "r2": r2, "adj_r2": adj_r2,
        "mse": mean_squared_error(y_clean, p(t))
    }
    if adj_r2 > best_adj_r2:
        best_adj_r2, best_deg = adj_r2, deg

poly = poly_results[best_deg]["poly"]
t_extra = np.arange(n, int(n * 1.5))
print(f"\nМНК: ступінь={best_deg}, R²={poly_results[best_deg]['r2']:.4f}, "
      f"MSE={poly_results[best_deg]['mse']:.6f}")


# R&D 3.2: Нелінійна модель Logistic + Linear
def nonlinear(t, L, k, t0, a, b):
    return L / (1 + np.exp(-k * (t - t0))) + a * t + b


scale = y_clean.max()
y_norm = y_clean / scale

try:
    popt, _ = curve_fit(
        nonlinear, t, y_norm,
        p0=[0.5, 0.05, n / 2, 0.001, 0.3],
        bounds=([0, 0, 0, 0, 0], [2, 1, n, 0.1, 2]),
        maxfev=20000
    )
    y_nl = nonlinear(t, *popt) * scale
    r2_nl = r2_score(y_clean, y_nl)
    nl_ok = True
    print(f"Нелінійна: R²={r2_nl:.4f}, MSE={mean_squared_error(y_clean, y_nl):.6f}")
except Exception as e:
    print(f"Нелінійна: {e}. Використовую МНК як fallback.")
    y_nl = poly(t)
    r2_nl = poly_results[best_deg]["r2"]
    popt = None
    nl_ok = False


# Група 2.5: Альфа-Бета-Гамма фільтр
def abg_filter(data, alpha=0.4, beta=0.05, gamma=0.01):
    x, v, a = data[0], 0.0, 0.0
    results = []
    for z in data:
        x_p = x + v + 0.5 * a
        v_p = v + a
        rk = z - x_p
        x = x_p + alpha * rk
        v = v_p + beta * rk
        a = a + gamma * rk
        results.append(x)
    return np.array(results)


df['Price_Filtered'] = abg_filter(df['Price_Clean'].values)
r2_filt = r2_score(y_clean, df['Price_Filtered'].values)
mse_filt = mean_squared_error(y_clean, df['Price_Filtered'].values)
print(f"ABG фільтр: R²={r2_filt:.4f}, MSE={mse_filt:.6f}")

# ВІЗУАЛІЗАЦІЯ
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(16, 13))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.28)
dates = df['Date']

# 1. Аномалії
ax1 = fig.add_subplot(gs[0, :])
ax1.scatter(dates, df['Price_Anom'], color='steelblue', alpha=0.5, s=15,
            label='Дані з аномаліями (PLN/UAH)')
ax1.scatter(dates.iloc[true_idx], df['Price_Anom'].iloc[true_idx],
            color='red', marker='D', s=60, label='Справжні аномалії')
ax1.scatter(dates.iloc[det_idx], df['Price_Anom'].iloc[det_idx],
            color='orange', marker='x', s=80, linewidths=2, label='Виявлені аномалії')
ax1.plot(dates, df['Price_Clean'], color='green', linewidth=1.5, label='Очищені дані')
ax1.set_title('R&D 3.1 — Sliding MAD: Виявлення аномалій у курсі PLN/UAH')
ax1.set_xlabel('Дата')
ax1.set_ylabel('Курс (UAH за 1 PLN)')
ax1.legend()
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

# 2. МНК
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(t, y_clean, color='steelblue', alpha=0.4, s=12, label='Очищені дані')
ax2.plot(t, poly(t), color='darkorange', linewidth=2, label=f'Поліном {best_deg} ст.')
ax2.plot(t_extra, poly(t_extra), '--', color='red', linewidth=2, label='Прогноз')
ax2.axvline(x=n - 1, linestyle=':', color='black')
ax2.set_title(f"МНК регресія (R² = {poly_results[best_deg]['r2']:.4f})")
ax2.set_xlabel('Спостереження')
ax2.set_ylabel('Курс (UAH за 1 PLN)')
ax2.legend()

# 3. Нелінійна
ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(t, y_clean, color='steelblue', alpha=0.4, s=12, label='Очищені дані')
ax3.plot(t, y_nl, color='purple', linewidth=2,
         label=f"{'Нелінійна' if nl_ok else 'МНК (fallback)'} (R²={r2_nl:.4f})")
if nl_ok and popt is not None:
    ax3.plot(t_extra, nonlinear(t_extra, *popt) * scale,
             '--', color='magenta', linewidth=2, label='Прогноз')
ax3.axvline(x=n - 1, linestyle=':', color='black')
ax3.set_title('R&D 3.2 — Logistic + Linear модель')
ax3.set_xlabel('Спостереження')
ax3.set_ylabel('Курс (UAH за 1 PLN)')
ax3.legend()

# 4. ABG фільтр
ax4 = fig.add_subplot(gs[2, :])
ax4.plot(dates, df['Price_Anom'], 'r.', alpha=0.3, label='Дані з аномаліями')
ax4.plot(dates, df['Price_Filtered'], 'b-', linewidth=2, label='ABG фільтр')
ax4.scatter(dates.iloc[det_idx], df['Price_Anom'].iloc[det_idx],
            color='black', marker='x', s=50, label='Виявлені аномалії')
ax4.set_title(f'Альфа-Бета-Гамма фільтр (R² = {r2_filt:.4f})')
ax4.set_xlabel('Дата')
ax4.set_ylabel('Курс (UAH за 1 PLN)')
ax4.legend()
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')

fig.suptitle(
    'Курс PLN/UAH (НБУ) — MAD детекція + МНК + Нелінійна + ABG фільтр\n'
    'Джерело: index.minfin.com.ua (парсинг HTML)',fontsize=13
)

plt.savefig('lab2_pln_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Графік збережено: lab2_pln_results.png")

mse_total = np.mean((df['Price'] - df['Price_Filtered']) ** 2)
print(f"MSE (оригінал vs ABG фільтр): {mse_total:.6f}")

plt.show()
