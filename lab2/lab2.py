import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# 1. БАЗОВІ ФУНКЦІЇ
def read_data(filename):
    x, y = [], []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['RPS']))
            y.append(float(row['CPU']))
    return x, y

def divided_diff_table(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:,0] = y
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    return coef

def newton_poly(coef, x_data, x):
    n = len(x_data) - 1 
    p = coef[0][n]
    for k in range(1, n+1):
        p = coef[0][n-k] + (x - x_data[n-k])*p
    return p

# Функція для методу Лагранжа (Додаткове завдання)
def lagrange_poly(x_data, y_data, x):
    result = 0.0
    for i in range(len(x_data)):
        term = y_data[i]
        for j in range(len(x_data)):
            if i != j:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result

x_base, y_base = read_data("C:/Users/ADMIN/Desktop/lab2/data.csv")

# Створюємо еталонну функцію на базі реальних 5 точок
f_true = CubicSpline(x_base, y_base)
x_plot_global = np.linspace(min(x_base), max(x_base), 500)
y_true_global = f_true(x_plot_global)

print("Розрахунок основної частини (Варіант 2)...")

# Генеруємо сітки на 10 та 20 вузлів
x_10_var = np.linspace(min(x_base), max(x_base), 10)
y_10_var = f_true(x_10_var)

x_20_var = np.linspace(min(x_base), max(x_base), 20)
y_20_var = f_true(x_20_var)

# Будуємо таблиці
diff_5 = divided_diff_table(x_base, y_base)
diff_10_var = divided_diff_table(x_10_var, y_10_var)
diff_20_var = divided_diff_table(x_20_var, y_20_var)

# Розрахунок значень
y_plot_5 = [newton_poly(diff_5, x_base, xi) for xi in x_plot_global]
y_plot_10_var = [newton_poly(diff_10_var, x_10_var, xi) for xi in x_plot_global]
y_plot_20_var = [newton_poly(diff_20_var, x_20_var, xi) for xi in x_plot_global]

# Похибки
err_5 = np.abs(y_true_global - y_plot_5)
err_10_var = np.abs(y_true_global - y_plot_10_var)
err_20_var = np.abs(y_true_global - y_plot_20_var)

# Побудова вікна для Варіанту 2
fig1, axs1 = plt.subplots(2, 2, figsize=(14, 10), num='Основне завдання (Варіант 2)')
fig1.suptitle('Аналіз інтерполяції поліномами Ньютона', fontsize=16)

axs1[0, 0].plot(x_plot_global, y_true_global, 'k--', label='Еталонна крива')
axs1[0, 0].plot(x_plot_global, y_plot_5, label='Ньютон (5 вузлів)', color='blue')
axs1[0, 0].scatter(x_base, y_base, color='black', s=50, label='Вузли')
axs1[0, 0].set_title('Інтерполяція (5 вузлів)')
axs1[0, 0].legend(); axs1[0, 0].grid(True)

axs1[0, 1].plot(x_plot_global, y_true_global, 'k--', label='Еталонна крива')
axs1[0, 1].plot(x_plot_global, y_plot_10_var, label='Ньютон (10 вузлів)', color='green')
axs1[0, 1].scatter(x_10_var, y_10_var, color='black', s=30)
axs1[0, 1].set_title('Інтерполяція (10 вузлів)')
axs1[0, 1].legend(); axs1[0, 1].grid(True)

axs1[1, 0].plot(x_plot_global, y_true_global, 'k--', label='Еталонна крива')
axs1[1, 0].plot(x_plot_global, y_plot_20_var, label='Ньютон (20 вузлів)', color='red')
axs1[1, 0].scatter(x_20_var, y_20_var, color='black', s=20)
axs1[1, 0].set_title('Інтерполяція (20 вузлів - Нестабільність)')
axs1[1, 0].set_ylim(-50, 300)
axs1[1, 0].legend(); axs1[1, 0].grid(True)

axs1[1, 1].plot(x_plot_global, err_5, label='Похибка (5 вузлів)', color='blue')
axs1[1, 1].plot(x_plot_global, err_10_var, label='Похибка (10 вузлів)', color='green')
axs1[1, 1].plot(x_plot_global, err_20_var, label='Похибка (20 вузлів)', color='red')
axs1[1, 1].set_title('Абсолютна похибка ε(x)')
axs1[1, 1].set_yscale('log')
axs1[1, 1].legend(); axs1[1, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

target_rps = 600
predicted_cpu = newton_poly(diff_5, x_base, target_rps)
print(f"Прогнозоване навантаження CPU при {target_rps} RPS: {predicted_cpu:.2f}%")
print("\nУВАГА: Закрийте перше вікно з графіками, щоб програма продовжила розрахунок дослідницької частини!")

plt.show() 

print("\nГенерація графіків дослідницької частини...")

# ДОСЛІДЖЕННЯ 1: Вплив кроку (Фіксований інтервал [50, 800], різна к-ть вузлів)
x_5 = np.linspace(50, 800, 5)
x_10 = np.linspace(50, 800, 10)
x_15 = np.linspace(50, 800, 15)

diff_5_res = divided_diff_table(x_5, f_true(x_5))
diff_10_res = divided_diff_table(x_10, f_true(x_10))
diff_15_res = divided_diff_table(x_15, f_true(x_15))

y_plot_5_res = [newton_poly(diff_5_res, x_5, xi) for xi in x_plot_global]
y_plot_10_res = [newton_poly(diff_10_res, x_10, xi) for xi in x_plot_global]
y_plot_15_res = [newton_poly(diff_15_res, x_15, xi) for xi in x_plot_global]

plt.figure('Дослідження 1: Вплив кількості вузлів', figsize=(10, 6))
plt.plot(x_plot_global, y_true_global, 'k--', label='Еталон (Сплайн)', linewidth=2)
plt.plot(x_plot_global, y_plot_5_res, label='n=5 вузлів', color='blue')
plt.plot(x_plot_global, y_plot_10_res, label='n=10 вузлів', color='green')
plt.plot(x_plot_global, y_plot_15_res, label='n=15 вузлів', color='red')
plt.title('1. Фіксований інтервал [50, 800], різна кількість вузлів')
plt.xlabel('RPS'); plt.ylabel('CPU (%)')
plt.ylim(-50, 300)
plt.legend(); plt.grid(True)

# ДОСЛІДЖЕННЯ 2: Вплив інтервалу (Фіксований крок h=100, різні інтервали)
x_int1 = np.arange(50, 351, 100) # 4 вузли
x_int2 = np.arange(50, 551, 100) # 6 вузлів
x_int3 = np.arange(50, 751, 100) # 8 вузлів

diff_int1 = divided_diff_table(x_int1, f_true(x_int1))
diff_int2 = divided_diff_table(x_int2, f_true(x_int2))
diff_int3 = divided_diff_table(x_int3, f_true(x_int3))

plt.figure('Дослідження 2: Вплив розміру інтервалу', figsize=(10, 6))
for x_nodes, diff_table, color, label in zip(
    [x_int1, x_int2, x_int3], [diff_int1, diff_int2, diff_int3], 
    ['blue', 'green', 'red'], ['Інтервал [50, 350]', 'Інтервал [50, 550]', 'Інтервал [50, 750]']):
    
    x_grid = np.linspace(min(x_nodes), max(x_nodes), 200)
    y_true_grid = f_true(x_grid)
    y_pred_grid = [newton_poly(diff_table, x_nodes, xi) for xi in x_grid]
    error = np.abs(y_true_grid - y_pred_grid)
    plt.plot(x_grid, error, color=color, label=f'{label} (Вузлів: {len(x_nodes)})')

plt.title('2. Фіксований крок h=100, змінний інтервал (Аналіз похибки)')
plt.xlabel('RPS'); plt.ylabel('Абсолютна похибка')
plt.legend(); plt.grid(True)

# ДОСЛІДЖЕННЯ 3: Аналіз ефекту Рунге (n=20 вузлів)
x_20 = np.linspace(50, 800, 20)
diff_20 = divided_diff_table(x_20, f_true(x_20))
y_plot_20 = [newton_poly(diff_20, x_20, xi) for xi in x_plot_global]
error_20 = np.abs(y_true_global - y_plot_20)

fig3, (ax1_3, ax2_3) = plt.subplots(2, 1, figsize=(10, 10), num='Дослідження 3: Ефект Рунге')
ax1_3.plot(x_plot_global, y_true_global, 'k--', label='Еталонна крива')
ax1_3.plot(x_plot_global, y_plot_20, 'r-', label='Ньютон (n=20 вузлів)')
ax1_3.scatter(x_20, f_true(x_20), color='black', s=20, label='Вузли')
ax1_3.set_title('3. Аналіз ефекту Рунге: Поліном високого степеня')
ax1_3.set_ylabel('CPU (%)'); ax1_3.set_ylim(-100, 400)
ax1_3.legend(); ax1_3.grid(True)

ax2_3.plot(x_plot_global, error_20, 'r-', label='Похибка ε(x) для n=20')
ax2_3.set_title('Зростання похибки на краях (Логарифмічна шкала)')
ax2_3.set_xlabel('RPS'); ax2_3.set_ylabel('Похибка')
ax2_3.set_yscale('log')
ax2_3.legend(); ax2_3.grid(True)

# ДОСЛІДЖЕННЯ 4: Порівняння Ньютона та Лагранжа
x_comp = np.linspace(50, 800, 8)
y_comp = f_true(x_comp)
diff_comp = divided_diff_table(x_comp, y_comp)

y_newton = np.array([newton_poly(diff_comp, x_comp, xi) for xi in x_plot_global])
y_lagrange = np.array([lagrange_poly(x_comp, y_comp, xi) for xi in x_plot_global])
diff_methods = np.abs(y_newton - y_lagrange)

fig4, (ax1_4, ax2_4) = plt.subplots(2, 1, figsize=(10, 10), num='Дослідження 4: Ньютон vs Лагранж')
ax1_4.plot(x_plot_global, y_newton, 'b-', linewidth=4, label='Метод Ньютона', alpha=0.5)
ax1_4.plot(x_plot_global, y_lagrange, 'r--', linewidth=2, label='Метод Лагранжа')
ax1_4.set_title('4. Порівняння методів (Криві ідеально накладаються)')
ax1_4.set_ylabel('CPU (%)')
ax1_4.legend(); ax1_4.grid(True)

ax2_4.plot(x_plot_global, diff_methods, 'g-', label='Різниця |Ньютон - Лагранж|')
ax2_4.set_title('Машинна похибка обчислень (близька до 10^-14)')
ax2_4.set_xlabel('RPS'); ax2_4.set_ylabel('Різниця')
ax2_4.legend(); ax2_4.grid(True)

print("Усі графіки дослідницької частини успішно згенеровані!")
plt.show() 