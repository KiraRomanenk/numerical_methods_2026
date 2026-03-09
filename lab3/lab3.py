import csv
import matplotlib.pyplot as plt

def read_csv(filename):
    """Зчитує середньомісячні температури з CSV файлу."""
    x = []
    y = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Пропускаємо заголовок (Month, Temp)
        for row in reader:
            if row:
                x.append(float(row[0]))
                y.append(float(row[1]))
    return x, y

# Метод найменших квадратів
def form_matrix(x, m):
    """Формування матриці A розміром (m+1) x (m+1)."""
    A = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            A[i][j] = sum(xi ** (i + j) for xi in x)
    return A

def form_vector(x, y, m):
    """Формування вектора вільних членів b розміром (m+1)."""
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(y[k] * (x[k] ** i) for k in range(len(x)))
    return b

def gauss_solve(A, b):
    """Розв'язок СЛАР методом Гауса з вибором головного елемента."""
    n = len(b)
    A = [row[:] for row in A]
    b = b[:]

    for k in range(n - 1):
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[max_row][k]):
                max_row = i

        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            if A[k][k] == 0: continue
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        suma = sum(A[i][j] * x_sol[j] for j in range(i + 1, n))
        if A[i][i] == 0:
            x_sol[i] = 0
        else:
            x_sol[i] = (b[i] - suma) / A[i][i]
    return x_sol

def polynomial(x, coef):
    """Обчислення значень полінома за знайденими коефіцієнтами."""
    return [sum(coef[i] * (xi ** i) for i in range(len(coef))) for xi in x]

def variance(y_true, y_approx):
    """Обчислення дисперсії (середньоквадратичного відхилення)."""
    n = len(y_true)
    return sum((y_true[i] - y_approx[i]) ** 2 for i in range(n)) / n

def main():
    # Автоматичне створення тестового CSV файлу згідно з прикладом
    csv_filename = "temperatures.csv"
    csv_data = """Month,Temp
1,-2\n2,0\n3,5\n4,10\n5,15\n6,20\n7,23\n8,22\n9,17\n10,10\n11,5\n12,0\n13,-10\n14,3\n15,7\n16,13\n17,19\n18,20\n19,22\n20,21\n21,18\n22,15\n23,10\n24,3"""
    with open(csv_filename, "w", encoding='utf-8') as f:
        f.write(csv_data)

    x, y = read_csv(csv_filename)

    max_degree = 10
    best_m = 1
    min_var = float('inf')
    best_coef = []

    print("-" * 40)
    print("Дисперсії для різних степенів полінома:")
    for m in range(1, max_degree + 1):
        A = form_matrix(x, m)
        b_vec = form_vector(x, y, m)
        coef = gauss_solve(A, b_vec)
        y_approx = polynomial(x, coef)
        var = variance(y, y_approx)
        print(f"Степінь m={m}: {var:.4f}")

        # Вибір оптимального степеня
        if var < min_var:
            min_var = var
            best_m = m
            best_coef = coef

    print("-" * 40)
    print(f"Оптимальний степінь полінома: {best_m} (Дисперсія: {min_var:.4f})")

    y_approx_best = polynomial(x, best_coef)

    # Екстраполяція
    x_future = [25, 26, 27]
    y_future = polynomial(x_future, best_coef)
    print("-" * 40)
    print("Екстраполяція (прогноз на наступні 3 місяці):")
    for m, t in zip(x_future, y_future):
        print(f"Місяць {m}: {t:.2f}°C")

    error_y = [abs(y[i] - y_approx_best[i]) for i in range(len(y))]
    print("-" * 40)
    print("Табуляція похибки апроксимації (|f(x) - phi(x)|):")
    for xi, err in zip(x, error_y):
        print(f"Місяць {xi:2.0f}: {err:.4f}")
    print("-" * 40)

    # Вікно 1
    plt.figure(1, figsize=(8, 5))
    plt.plot(x, y, 'o', label='Фактичні дані', color='blue')
    plt.plot(x, y_approx_best, '-', label=f'Апроксимація (степінь {best_m})', color='red')
    plt.title('1. Графік апроксимації та фактичних даних')
    plt.xlabel('Місяць')
    plt.ylabel('Температура (°C)')
    plt.grid(True)
    plt.legend()

    # Вікно 2
    plt.figure(2, figsize=(8, 5))
    plt.plot(x, error_y, '-x', color='purple', label='Абсолютна похибка')
    plt.title('2. Графік похибки апроксимації')
    plt.xlabel('Місяць')
    plt.ylabel('Похибка')
    plt.grid(True)
    plt.legend()

    # Вікно 3
    plt.figure(3, figsize=(8, 5))
    plt.plot(x[-5:], y[-5:], 'o', label='Останні фактичні дані', color='blue')
    plt.plot(x_future, y_future, 's--', label='Прогноз (наступні 3 міс.)', color='green')
    plt.title('3. Екстраполяція: прогноз температури')
    plt.xlabel('Місяць')
    plt.ylabel('Температура (°C)')
    plt.grid(True)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()