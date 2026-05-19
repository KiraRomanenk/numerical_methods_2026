import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(X):
    return 100 * (X[0]**2 - X[1])**2 + (X[0] - 1)**2

def f1(x1, x2): 
    return x1**2 + x2**2 - 4

def f2(x1, x2): 
    return x1 - x2**2 + 1

def objective_function(X):
    # Цільова функція: сума квадратів рівнянь
    return f1(X[0], X[1])**2 + f2(X[0], X[1])**2

def exploratory_search(X_base, delta, q, eps1, func):
    X = np.copy(X_base)
    n = len(X)
    for i in range(n):
        f_old = func(X)
        X[i] += delta[i]
        if func(X) < f_old:
            continue
        X[i] -= 2 * delta[i]
        if func(X) < f_old:
            continue
        X[i] += delta[i] 
        
        delta[i] /= q
        if delta[i] >= eps1:
            temp_X = np.copy(X)
            temp_X[i] += delta[i]
            if func(temp_X) < f_old: 
                X[i] = temp_X[i]
            else:
                temp_X[i] -= 2 * delta[i]
                if func(temp_X) < f_old: 
                    X[i] = temp_X[i]
    return X, delta

def hooke_jeeves(X0, delta0, q, p, eps1, eps2, func):
    X_old = np.copy(X0)
    delta = np.copy(delta0)
    trajectory = [np.copy(X_old)]
    
    X_new, delta = exploratory_search(X_old, delta, q, eps1, func)

    while True:
        trajectory.append(np.copy(X_new))
        
        # Умови зупинки
        if (np.linalg.norm(X_new - X_old) < eps1 and 
            abs(func(X_new) - func(X_old)) < eps2):
            break
        if np.array_equal(X_new, X_old):
            break


        X_pattern = X_new + p * (X_new - X_old)
        X_after_pattern, _ = exploratory_search(X_pattern, delta, 1.0, 0, func)

        X_old = np.copy(X_new)
        if func(X_after_pattern) < func(X_new):
            X_new = np.copy(X_after_pattern)
        else:
            X_new, delta = exploratory_search(X_old, delta, q, eps1, func)

    return X_new, trajectory

if __name__ == "__main__":
    q = 2.0
    p = 2.0 
    eps1 = 1e-5
    eps2 = 1e-5

    print("--- РОЗВ'ЯЗОК СИСТЕМИ РІВНЯНЬ ---")
    X0_sys = np.array([-2.0, 1.0])
    delta0_sys = np.array([0.2, 0.2])
    
    result, path = hooke_jeeves(X0_sys, delta0_sys, q, p, eps1, eps2, objective_function)

    print(f"Розв'язок: x1 = {result[0]:.6f}, x2 = {result[1]:.6f}")
    print(f"Значення цільової функції (похибка): {objective_function(result):.10e}")
    print(f"Кількість кроків: {len(path)}")

    with open(r"C:\Users\ADMIN\Desktop\trajectory.txt", "w") as f:
        for point in path:
            f.write(f"{point[0]:.6f}, {point[1]:.6f}\n")

    # Побудова графіка
    x_range = np.linspace(-3, 3, 400)
    y_range = np.linspace(-3, 3, 400)
    X1, X2 = np.meshgrid(x_range, y_range)

    plt.figure(figsize=(10, 8))
    plt.contour(X1, X2, f1(X1, X2), [0], colors='blue', linewidths=2)
    plt.contour(X1, X2, f2(X1, X2), [0], colors='red', linewidths=2)

    path_coords = np.array(path)
    plt.plot(path_coords[:, 0], path_coords[:, 1], 'g--o', markersize=4, label='Траєкторія')
    plt.plot(X0_sys[0], X0_sys[1], 'ko', label='Старт')
    plt.plot(result[0], result[1], 'r*', markersize=12, label='Розв\'язок')

    plt.title('Графік рівнянь та траєкторія методу Хука-Дживса')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(['f1(x)=0 (коло)', 'f2(x)=0 (парабола)', 'Траєкторія', 'Старт', 'Розв\'язок'])
    plt.grid(True)
    plt.show()