import numpy as np
import matplotlib.pyplot as plt

def f(x, y): return x - y # Диференціальне рівняння y' = f(x, y)
def exact_sol(x): return x - 1 + 2 * np.exp(-x) #Точний (аналітичний) розв'язок для порівняння

a, b, y0 = 0, 2, 1.0
h_fix = 0.01  # Заданий крок 10^-2 за умовою (П. 6)
eps = 1e-4

def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h * k1/2)
    k3 = f(x + h/2, y + h * k2/2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

def adams_pc2_step(x_prev, y_prev, x_n, y_n, h, eps_iter=1e-5):
    f_n = f(x_n, y_n)
    f_prev = f(x_prev, y_prev)
    y_pr = y_n + (h / 2) * (3 * f_n - f_prev) # Прогноз
    
    y_kor = y_pr
    while True:
        y_kor_new = y_n + (h / 2) * (f(x_n + h, y_kor) + f_n) # Корекція
        if abs(y_kor_new - y_kor) < eps_iter:
            break
        y_kor = y_kor_new
    return y_kor, y_pr

def adams_fixed(a, b, y0, h):
    x_vals = np.arange(a, b + h, h)
    y_vals = np.zeros(len(x_vals))
    y_vals[0] = y0
    y_vals[1] = rk4_step(x_vals[0], y_vals[0], h)
    err_est = [0, 0]
    
    for i in range(1, len(x_vals) - 1):
        y_kor, y_pr = adams_pc2_step(x_vals[i-1], y_vals[i-1], x_vals[i], y_vals[i], h)
        y_vals[i+1] = y_kor
        err_est.append(abs(y_kor - y_pr))
    return x_vals, y_vals, np.array(err_est)

def adams_auto(a, b, y0, h, eps):
    x_vals, y_vals, h_vals = [a], [y0], [h]
    x, y = a, y0
    y_prev, x_prev = None, None
    
    while x < b:
        if x_prev is None: 
            y_next = rk4_step(x, y, h)
            err_est = 0
        else:
            y_kor, y_pr = adams_pc2_step(x_prev, y_prev, x, y, h)
            y_next = y_kor
            err_est = abs(y_kor - y_pr)
            
        if err_est > eps and x_prev is not None:
            h /= 2
            x_prev = None 
        else:
            x_prev, y_prev = x, y
            x += h
            y = y_next
            x_vals.append(x)
            y_vals.append(y)
            h_vals.append(h)
            if err_est <= eps / 10 and x_prev is not None:
                h *= 2
                x_prev = None
    return np.array(x_vals), np.array(y_vals), np.array(h_vals)

def rk4_fixed(a, b, y0, h):
    x_vals = np.arange(a, b + h, h)
    y_vals = np.zeros(len(x_vals))
    y_vals[0] = y0
    for i in range(len(x_vals) - 1):
        y_vals[i+1] = rk4_step(x_vals[i], y_vals[i], h)
    return x_vals, y_vals

def rk4_auto(a, b, y0, h, eps):
    x_vals, y_vals, h_vals, runge_errs = [a], [y0], [h], [0]
    x, y = a, y0
    
    while x < b:
        y_h = rk4_step(x, y, h)
        y_half_1 = rk4_step(x, y, h/2)
        y_half_2 = rk4_step(x + h/2, y_half_1, h/2)
        
        err = (16 / 15) * abs(y_h - y_half_2)
        
        if err > eps:
            h /= 2
        else:
            x += h
            y = y_h
            x_vals.append(x)
            y_vals.append(y)
            h_vals.append(h)
            runge_errs.append(err)
            if err <= eps / 32:
                h *= 2
    return np.array(x_vals), np.array(y_vals), np.array(h_vals), np.array(runge_errs)

x_ad, y_ad, err_est_ad = adams_fixed(a, b, y0, h_fix)
x_rk, y_rk = rk4_fixed(a, b, y0, h_fix)

x_ad_a, y_ad_a, h_ad_a = adams_auto(a, b, y0, h_fix, eps)
x_rk_a, y_rk_a, h_rk_a, err_runge = rk4_auto(a, b, y0, 0.1, eps)

# П. 7: Дослідження залежності похибки від кроку
h_test = [0.1, 0.05, 0.025, 0.01]
max_errs_rk = []
for ht in h_test:
    xt, yt = rk4_fixed(a, b, y0, ht)
    max_errs_rk.append(np.max(np.abs(yt - exact_sol(xt))))

fig = plt.figure(figsize=(16, 12))

# П.3 та П.4: Похибки Адамса
ax1 = fig.add_subplot(231)
ax1.plot(x_ad, np.abs(y_ad - exact_sol(x_ad)), 'r-', label='Точна похибка')
ax1.plot(x_ad, err_est_ad, 'g--', label='Оцінка (y_kor - y_pr)')
ax1.set_title('П. 3, 4: Похибки Адамса')
ax1.legend(); ax1.grid()

# П.5: Авто-крок Адамса
ax2 = fig.add_subplot(232)
ax2.step(x_ad_a, h_ad_a, 'm-')
ax2.set_title('П. 5: Зміна кроку h(x) Адамс')
ax2.grid()

# П.7: Похибка РК4 та залежність від h
ax3 = fig.add_subplot(233)
ax3.plot(x_rk, np.abs(y_rk - exact_sol(x_rk)), 'b-')
ax3.set_title(f'П. 7: Точна похибка РК4 (h={h_fix})')
ax3.grid()

ax4 = fig.add_subplot(234)
ax4.loglog(h_test, max_errs_rk, 'o-')
ax4.set_title('П. 7: Залежність max похибки від h (РК4)')
ax4.set_xlabel('Крок h'); ax4.set_ylabel('Max похибка')
ax4.grid()

# П.8: Оцінка похибки за Рунге
ax5 = fig.add_subplot(235)
ax5.plot(x_rk_a, np.abs(y_rk_a - exact_sol(x_rk_a)), 'r-', label='Точна')
ax5.plot(x_rk_a, err_runge, 'b--', label='За Рунге')
ax5.set_title('П. 8: Похибка РК4 (Рунге vs Точна)')
ax5.legend(); ax5.grid()

# П.9: Авто-крок РК4
ax6 = fig.add_subplot(236)
ax6.step(x_rk_a, h_rk_a, 'k-')
ax6.set_title('П. 9: Зміна кроку h(x) РК4')
ax6.grid()

# Вивід таблиці результатів у консоль (для фіксованого кроку)
print("\n" + "="*95)
print("Таблиця результатів для фіксованого кроку h = 0.01 (вивід кожного 20-го вузла)")
print("="*95)
print(f"{'x':<5} | {'Точне y(x)':<13} | {'y (Адамс)':<13} | {'Похибка Адамса':<16} | {'y (РК4)':<13} | {'Похибка РК4':<16}")
print("-" * 95)

# Проходимо по масиву з кроком 20, щоб виводити значення рівномірно
for i in range(0, len(x_ad), 20):
    x_val = x_ad[i]
    y_ex = exact_sol(x_val)
    
    y_a = y_ad[i]
    err_a = abs(y_a - y_ex)
    
    y_r = y_rk[i]
    err_r = abs(y_r - y_ex)

    print(f"{x_val:<5.2f} | {y_ex:<13.6f} | {y_a:<13.6f} | {err_a:<16.2e} | {y_r:<13.6f} | {err_r:<16.2e}")

print("="*95 + "\n")

plt.tight_layout()
plt.show()