import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

# Трансцендентна функція
def F(x): 
    return math.exp(x) - 4*x - 2

def dF(x): 
    return math.exp(x) - 4

def d2F(x): 
    return math.exp(x)

def check_stop(x_next, x_prev, eps):
    return abs(F(x_next)) < eps and abs(x_next - x_prev) < eps

# Методи уточнення

def newton_method(x0, eps):
    it, x_prev = 0, x0
    while it < 500:
        it += 1
        df = dF(x_prev)
        if abs(df) < 1e-15: x_prev += 0.1; continue
        x_next = x_prev - F(x_prev) / df
        if check_stop(x_next, x_prev, eps): return x_next, it
        x_prev = x_next
    return x_prev, it

def secant_method(x_prev, x_curr, eps):
    it = 0
    while it < 500:
        it += 1
        f_curr, f_prev = F(x_curr), F(x_prev)
        if abs(f_curr - f_prev) < 1e-15: break
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        if check_stop(x_next, x_curr, eps): return x_next, it
        x_prev, x_curr = x_curr, x_next
    return x_curr, it

def parabolic_method(x0, x1, x2, eps):
    it = 0
    while it < 500:
        it += 1
        f0, f1, f2 = F(x0), F(x1), F(x2)
        f10 = (f1 - f0) / (x1 - x0)
        f21 = (f2 - f1) / (x2 - x1)
        f210 = (f21 - f10) / (x2 - x0)
        w = f21 + (x2 - x1) * f210
        det = cmath.sqrt(w**2 - 4 * f2 * f210)
        denom = w + det if abs(w + det) > abs(w - det) else w - det
        if abs(denom) < 1e-15: break
        delta = -2 * f2 / denom
        x_next = x2 + delta.real
        if check_stop(x_next, x2, eps): return x_next, it
        x0, x1, x2 = x1, x2, x_next
    return x2, it

# Алгебраїчні рівняння

def horner_eval(a, x):
    m = len(a) - 1
    b = [0]*(m+1)
    b[m] = a[m]
    for i in range(m - 1, -1, -1): b[i] = a[i] + x * b[i + 1]
    c = [0]*(m+1)
    c[m] = b[m]
    for i in range(m - 1, 0, -1): c[i] = b[i] + x * c[i + 1]
    return b[0], c[1]

def newton_horner(a, x0, eps):
    it, x_prev = 0, x0
    while it < 500:
        it += 1
        f_val, df_val = horner_eval(a, x_prev)
        if abs(df_val) < 1e-15: x_prev += 0.1; continue
        x_next = x_prev - f_val / df_val
        if abs(x_next - x_prev) < eps: return x_next, it
        x_prev = x_next
    return x_prev, it

def lin_method(a, alpha0, beta0, eps):
    it, m = 0, len(a) - 1
    p, q = -2 * alpha0, alpha0**2 + beta0**2
    while it < 500:
        it += 1
        b = [0]*(m+1)
        b[m] = a[m]
        b[m-1] = a[m-1] - p * b[m]
        for i in range(m-2, 1, -1):
            b[i] = a[i] - p * b[i+1] - q * b[i+2]
        if abs(b[2]) < 1e-15: break
        q_new, p_new = a[0] / b[2], (a[1] - (a[0] / b[2]) * b[3]) / b[2]
        al_n = -p_new / 2
        det = p_new**2 - 4 * q_new
        be_n = math.sqrt(abs(det)) / 2 if det < 0 else 0
        if abs(al_n - alpha0) < eps and abs(be_n - beta0) < eps: return al_n, be_n, it
        p, q, alpha0, beta0 = p_new, q_new, al_n, be_n
    return alpha0, beta0, it

if __name__ == "__main__":
    EPS = 1e-10
    
    # Табуляція 
    approx = []
    x, h = -1.0, 0.1
    while x <= 3.0:
        x_n = round(x + h, 2)
        if F(x) * F(x_n) < 0:
            approx.append(((x + x_n)/2, "зростання" if F(x_n) > F(x) else "спадання"))
        x = x_n

    print(f"{'Метод':<25} | {'Результат':<18} | {'Кількість ітерацій'}")
    print("-" * 65)

    for r0, beh in approx:
        print(f"\n--- Аналіз кореня ({beh}) ---")
        for name, func in [("Метод Ньютона", newton_method), ("Метод хорд", lambda r, e: secant_method(r-0.05, r+0.05, e)), ("Метод парабол", lambda r, e: parabolic_method(r-0.1, r, r+0.1, e))]:
            res, it = func(r0, EPS)
            print(f"{name:<25} | {res:<18.10f} | Ітерацій: {it}")

    print("\n" + "="*65 + "\nАЛГЕБРАЇЧНЕ РІВНЯННЯ (x^3 - 3x^2 + 4x - 2 = 0)\n" + "="*65)
    coeffs = [-2.0, 4.0, -3.0, 1.0]
    res_h, it_h = newton_horner(coeffs, 0.5, EPS)
    print(f"{'Ньютон (Горнер)':<25} | {res_h:<18.10f} | Ітерацій: {it_h}")
    al, be, it_l = lin_method(coeffs, 0.8, 0.8, EPS)
    print(f"{'Метод Ліна':<25} | {al:.4f} ± {be:.4f}i | Ітерацій: {it_l}")

    # Побудова графіка
    x_val = np.linspace(-0.5, 2.5, 400)
    y_val = coeffs[3]*x_val**3 + coeffs[2]*x_val**2 + coeffs[1]*x_val + coeffs[0]
    plt.figure(figsize=(8, 5))
    plt.plot(x_val, y_val, label='f(x) = x³-3x²+4x-2', color='blue')
    plt.axhline(0, color='black', lw=1); plt.axvline(0, color='black', lw=1)
    plt.scatter([1.0], [0], color='red', label='Дійсний корінь (x=1)')
    plt.title("Графік алгебраїчного рівняння"); plt.grid(True); plt.legend()
    plt.show()

   # Побудова графіка для трансцендентної функції
x_trans = np.linspace(-1, 3, 500)
y_trans = [F(val) for val in x_trans]

plt.figure(figsize=(10, 6))
plt.plot(x_trans, y_trans, label=r'$f(x) = e^x - 4x - 2$', color='green', linewidth=2)

plt.axhline(0, color='black', linestyle='--', linewidth=1)

for r0, _ in approx:
    root, _ = newton_method(r0, EPS)
    plt.scatter(root, 0, color='red', s=50, zorder=5, label='Корінь' if 'Корінь' not in plt.gca().get_legend_handles_labels()[1] else "")

# Налаштування вигляду
plt.title("Табуляція та знаходження коренів трансцендентної функції", fontsize=14)
plt.xlabel("x")
plt.ylabel("F(x)")
plt.grid(True, which='both', linestyle=':', alpha=0.7)

plt.legend()
plt.ylim(-5, 10) 
plt.show()