import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    """Функція, що описує інтенсивність навантаження."""
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

# межі інтегрування
a, b = 0, 24

I_0, _ = quad(f, a, b, epsabs=1e-14, epsrel=1e-14)

def simpson_integral(func, a, b, N):
    """Обчислення інтегралу методом Сімпсона при заданому числі розбиттів N."""
    if N % 2 != 0:
        N += 1
    
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = func(x)

    integral = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return integral

N_values = np.arange(10, 1002, 2)
errors = np.zeros(len(N_values))

N_opt = None
eps_opt = None
target_eps = 1e-12

for i, N in enumerate(N_values):
    I_N = simpson_integral(f, a, b, N)
    err = abs(I_N - I_0)
    errors[i] = err
    
    if err <= target_eps and N_opt is None:
        N_opt = N
        eps_opt = err

if N_opt is None:
    N_opt = N_values[-1]
    eps_opt = errors[-1]

N0_approx = max(8, round((N_opt / 10) / 8) * 8)
N0 = int(N0_approx)

I_N0 = simpson_integral(f, a, b, N0)
eps0 = abs(I_N0 - I_0)

I_N0_2 = simpson_integral(f, a, b, N0 // 2)
I_R = I_N0 + (I_N0 - I_N0_2) / 15.0 # метод рунге-ромбюерга
epsR = abs(I_R - I_0)

I_N0_4 = simpson_integral(f, a, b, N0 // 4)
num_p = I_N0_4 - I_N0_2
den_p = I_N0_2 - I_N0

if den_p != 0 and (num_p / den_p) > 0:
    p = (1 / np.log(2)) * np.log(abs(num_p / den_p)) # метод ейткена (порядок точності)
else:
    p = float('nan')

denom_E = 2 * I_N0_2 - (I_N0 + I_N0_4)
if denom_E != 0:
    I_E = (I_N0_2**2 - I_N0 * I_N0_4) / denom_E
else:
    I_E = I_N0 # Fallback, якщо знаменник 0
epsE = abs(I_E - I_0)

class AdaptiveSimpson:
    """Клас для розрахунку адаптивного інтегралу з підрахунком викликів функції."""
    def __init__(self, func):
        self.func = func
        self.evals = 0
        self.cache = {}

    def evaluate(self, x):
        if x not in self.cache:
            self.cache[x] = self.func(x)
            self.evals += 1
        return self.cache[x]

    def integrate(self, a, b, delta):
        h = b - a
        c = (a + b) / 2
        
        fa = self.evaluate(a)
        fc = self.evaluate(c)
        fb = self.evaluate(b)
        
        I1 = (h / 6) * (fa + 4 * fc + fb)
        
        d = (a + c) / 2
        e = (c + b) / 2
        fd = self.evaluate(d)
        fe = self.evaluate(e)
        
        I2_left = (h / 12) * (fa + 4 * fd + fc)
        I2_right = (h / 12) * (fc + 4 * fe + fb)
        I2 = I2_left + I2_right
        
        if abs(I1 - I2) <= delta:
            return I2
        else:
            return self.integrate(a, c, delta / 2) + self.integrate(c, b, delta / 2)

deltas_test = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
adaptive_results = []

for d_val in deltas_test:
    ad_solver = AdaptiveSimpson(f)
    I_ad = ad_solver.integrate(a, b, d_val)
    err_ad = abs(I_ad - I_0)
    adaptive_results.append((d_val, I_ad, err_ad, ad_solver.evals))


print(" ЛАБОРАТОРНА РОБОТА №5: Квадратурні формули Сімпсона")
print("-"*60)
print(f"Точне значення інтегралу I_0: {I_0:.15f}")
print(f"Оптимальне число розбиттів N_opt для eps=1e-12: {N_opt}")
print(f"Похибка при N_opt (eps_opt): {eps_opt:.3e}\n")

print(f"Вибране початкове N0 (кратне 8, ~N_opt/10): {N0}")
print(f"Значення I(N0): {I_N0:.15f}")
print(f"Похибка eps0: {eps0:.3e}\n")

print("Метод Рунге-Ромберга")
print(f"Уточнене значення I_R: {I_R:.15f}")
print(f"Похибка epsR: {epsR:.3e}")
print(f"Зменшення похибки порівняно з eps0 у {eps0/epsR:.1f} разів\n" if epsR!=0 else "Точно співпало\n")

print("Метод Ейткена")
print(f"Оцінений порядок методу p: {p:.4f} (Теоретичний для Сімпсона: ~4.0)")
print(f"Уточнене значення I_E: {I_E:.15f}")
print(f"Похибка epsE: {epsE:.3e}")
print(f"Зменшення похибки порівняно з eps0 у {eps0/epsE:.1f} разів\n" if epsE!=0 else "Точно співпало\n")

print("Адаптивний алгорит-")
print(f"{'Задана δ':>10} | {'Похибка (eps)':>15} | {'Викликів f(x)':>15}")
print("-" * 46)
for d_val, _, err_ad, evals in adaptive_results:
    print(f"{d_val:10.0e} | {err_ad:15.3e} | {evals:15d}")
print("="*60)


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
x_vals = np.linspace(a, b, 1000)
plt.plot(x_vals, f(x_vals), color='blue', label='$f(x)$')
plt.title('Графік функції навантаження на сервер')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(N_values, errors, color='red', label='Похибка $\epsilon(N)$')
plt.axhline(target_eps, color='green', linestyle='--', label=f'Задана точність {target_eps}')
if N_opt is not None:
    plt.axvline(N_opt, color='black', linestyle=':', label=f'$N_{{opt}} = {N_opt}$')
plt.yscale('log')
plt.title('Залежність похибки від кількості розбиттів N')
plt.xlabel('Число розбиттів N')
plt.ylabel('Похибка $|I(N) - I_0|$ (логарифмічна шкала)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()