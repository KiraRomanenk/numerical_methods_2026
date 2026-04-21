import random
import math

def generate_random_matrix(n):
    return [[random.uniform(-100, 100) for _ in range(n)] for _ in range(n)]

def write_matrix_to_file(matrix, filename):
    with open(filename, 'w') as f:
        for row in matrix:
            f.write(" ".join(f"{val:.8f}" for val in row) + "\n")

def read_matrix_from_file(filename):
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            matrix.append([float(x) for x in line.split()])
    return matrix

def write_vector_to_file(vector, filename):
    with open(filename, 'w') as f:
        f.write("\n".join(f"{val:.8f}" for val in vector))

def read_vector_from_file(filename):
    vector = []
    with open(filename, 'r') as f:
        for line in f:
            vector.append(float(line.strip()))
    return vector

def multiply_matrix_vector(A, X):
    n = len(A)
    B = [0.0] * n
    for i in range(n):
        B[i] = sum(A[i][j] * X[j] for j in range(n))
    return B

def subtract_vectors(V1, V2):
    return [v1 - v2 for v1, v2 in zip(V1, V2)]

def add_vectors(V1, V2):
    return [v1 + v2 for v1, v2 in zip(V1, V2)]

def vector_norm(V):
    return max(abs(v) for v in V)


def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        U[i][i] = 1.0

    for k in range(n):
        for i in range(k, n):
            sum_l_u = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - sum_l_u
        for i in range(k + 1, n):
            sum_l_u = sum(L[k][j] * U[j][i] for j in range(k))
            if L[k][k] == 0:
                raise ValueError("Нульовий елемент на діагоналі L. Розклад неможливий без перестановки рядків.")
            U[k][i] = (A[k][i] - sum_l_u) / L[k][k]

    return L, U

def solve_lu(L, U, B):
    n = len(L)
    Z = [0.0] * n
    X = [0.0] * n

    for k in range(n):
        sum_lz = sum(L[k][j] * Z[j] for j in range(k))
        Z[k] = (B[k] - sum_lz) / L[k][k]

    for k in range(n - 1, -1, -1):
        sum_ux = sum(U[k][j] * X[j] for j in range(k + 1, n))
        X[k] = Z[k] - sum_ux

    return X

def main():
    n = 100
    eps_target = 10**(-14)
    
    print(f"Лабораторна робота №7: LU-розклад (n={n})\n")
    print("1. Генерація матриці A та вектора B...")
    A_generated = generate_random_matrix(n)
    write_matrix_to_file(A_generated, "matrix_A.txt")

    X_exact = [2.5] * n
    B_generated = multiply_matrix_vector(A_generated, X_exact)
    write_vector_to_file(B_generated, "vector_B.txt")

    print("2. Зчитування даних з файлів та виконання LU-розкладу...")
    A = read_matrix_from_file("matrix_A.txt")
    B = read_vector_from_file("vector_B.txt")
    
    L, U = lu_decomposition(A)
    write_matrix_to_file(L, "matrix_L.txt")
    write_matrix_to_file(U, "matrix_U.txt")

    print("3. Первинний розв'язок системи рівнянь AX=B...")
    X_0 = solve_lu(L, U, B)

    AX_0 = multiply_matrix_vector(A, X_0)
    R_0 = subtract_vectors(AX_0, B)
    eps_initial = vector_norm(R_0)
    print(f"4. Початкова похибка (нев'язка) розв'язку: {eps_initial:.4e}")

    print("\n5. Початок ітераційного уточнення...")
    X_current = list(X_0)
    iteration = 0

    while True:
        iteration += 1
        
        AX_curr = multiply_matrix_vector(A, X_current)
        R = subtract_vectors(B, AX_curr)
        
        delta_X = solve_lu(L, U, R)
        
        X_current = add_vectors(X_current, delta_X)

        norm_delta_X = vector_norm(delta_X)
        
        AX_new = multiply_matrix_vector(A, X_current)
        norm_R_new = vector_norm(subtract_vectors(AX_new, B))
        
        print(f"   Ітерація {iteration}: ||delta_X|| = {norm_delta_X:.4e}, ||AX-B|| = {norm_R_new:.4e}")
        
        if norm_delta_X <= eps_target and norm_R_new <= eps_target:
            print("\nУмови закінчення ітераційної процедури виконано!")
            break
            
        # Захист від нескінченного циклу (у випадку погано обумовленої матриці)
        if iteration > 50:
            print("\nДосягнуто ліміт у 50 ітерацій. Подальше уточнення неможливе через машинну точність.")
            break

    diff_from_exact = vector_norm(subtract_vectors(X_current, X_exact))
    print(f"\nФінальна точність відносно еталонного розв'язку (2.5): {diff_from_exact:.4e}")
    print("Усі текстові файли з матрицями (A, L, U) та вектором (B) успішно створені в директорії скрипта.")

if __name__ == "__main__":
    main()