import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp

def solve_mco():
    alternatives = ["Nova Poshta", "Ukrposhta", "Meest", "Delivery Auto", "SAT"]
    benefit = [False, False, True, True, True]

    matrix = np.array([
        [80, 24, 98, 478, 10],
        [45, 72, 85,  35,  6],
        [49, 48, 90,  80,  8],
        [45, 48, 85,   5,  5],
        [55, 48, 88,   3,  5],
    ], dtype=float)

    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

    norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    v = norm_matrix * weights

    i_best = np.where(benefit, v.max(axis=0), v.min(axis=0))
    i_worst = np.where(benefit, v.min(axis=0), v.max(axis=0))

    d_best = np.sqrt(((v - i_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((v - i_worst) ** 2).sum(axis=1))
    topsis_ci = d_worst / (d_best + d_worst)

    df = pd.DataFrame({
        "Кур'єр": alternatives,
        "TOPSIS_Ci": topsis_ci
    }).sort_values(by="TOPSIS_Ci", ascending=True)

    print("=== РЕЗУЛЬТАТИ ОЦІНЮВАННЯ КУР'ЄРІВ (TOPSIS) ===")
    print(df.sort_values(by="TOPSIS_Ci", ascending=False).to_string(index=False))
    print("-" * 50)

    return df

def solve_lp():
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return None, None, None

    x1 = solver.NumVar(0, solver.infinity(), 'x1')
    x2 = solver.NumVar(0, solver.infinity(), 'x2')

    solver.Add(1.5 * x1 + 2 * x2 <= 12, "C1")
    solver.Add(1 * x1 + 2 * x2 <= 8, "C2")
    solver.Add(4 * x1 <= 16, "C3")
    solver.Add(4 * x2 <= 12, "C4")

    solver.Maximize(2 * x1 + 2 * x2)
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        opt_x1 = x1.solution_value()
        opt_x2 = x2.solution_value()
        opt_q = -2 * opt_x1 - 2 * opt_x2

        print("=== РОЗВ'ЯЗОК ЗЛП (Лекція 6) ===")
        print(f"Мінімум цільової функції (Q) = {opt_q}")
        print(f"X1 = {opt_x1}")
        print(f"X2 = {opt_x2}")
        print("-" * 50)
        return opt_x1, opt_x2, opt_q
    return None, None, None

def main():
    df_mco = solve_mco()
    solve_lp()

    fig, ax1 = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title("Результати Аналізу: TOPSIS")

    bars = ax1.barh(df_mco["Кур'єр"], df_mco["TOPSIS_Ci"], color='skyblue', edgecolor='black')
    ax1.set_title("Рейтинг кур'єрських служб (TOPSIS Ci)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Коефіцієнт близькості до ідеалу (ближче до 1 = краще)")
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    for bar in bars:
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.3f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("topsis_rating.png", dpi=300, bbox_inches='tight')
    print("Графік успішно збережено у файл 'topsis_rating.png'")
    plt.show()

if __name__ == "__main__":
    main()