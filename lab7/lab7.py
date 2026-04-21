import random
import math

def generate_matrix(n):
    A = [[random.uniform(1, 5) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        row_sum = sum(abs(A[i][j]) for j in range(n) if i != j)
        A[i][i] = row_sum + random.uniform(1, 10)
    return A

def write_matrix(filename, A):
    with open(filename, "w") as f:
        for row in A:
            f.write(" ".join(map(str, row)) + "\n")

def read_matrix(filename):
    with open(filename, "r") as f:
        return [list(map(float, line.split())) for line in f]

def generate_B(A, x_true):
    n = len(A)
    B = []
    for i in range(n):
        s = sum(A[i][j] * x_true[j] for j in range(n)) 
        B.append(s)
    return B

def write_vector(filename, v):
    with open(filename, "w") as f:
        for x in v:
            f.write(f"{x}\n")

def read_vector(filename):
    with open(filename, "r") as f:
        return [float(line.strip()) for line in f]

def norm_vector(v):
    # Обчислення норми вектора (max abs) 
    return max(abs(x) for x in v)

def norm_matrix(A):
    return max(sum(abs(x) for x in row) for row in A)

def mat_vec(A, x):
    n = len(A)
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]

def get_residual_norm(A, x, B):
    # Обчислення норми нев'язки ||AX - B|| 
    ax = mat_vec(A, x)
    res = [ax[i] - B[i] for i in range(len(B))]
    return norm_vector(res)

def simple_iteration(A, B, x0, eps=1e-14, max_iter=10000):
    n = len(A)
    tau = 1.0 / norm_matrix(A) 
    x = x0[:]
    for k in range(max_iter):
        Ax = mat_vec(A, x)
        x_new = [x[i] - tau * (Ax[i] - B[i]) for i in range(n)] 
        if norm_vector([x_new[i] - x[i] for i in range(n)]) < eps: 
            return x_new, k + 1
        x = x_new
    return x, max_iter

def jacobi(A, B, x0, eps=1e-14, max_iter=10000):
    n = len(A)
    x = x0[:]
    for k in range(max_iter):
        x_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (B[i] - s) / A[i][i]
        if norm_vector([x_new[i] - x[i] for i in range(n)]) < eps:
            return x_new, k + 1
        x = x_new
    return x, max_iter

def seidel(A, B, x0, eps=1e-14, max_iter=10000):
    n = len(A)
    x = x0[:]
    for k in range(max_iter):
        x_old = x[:]
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (B[i] - s1 - s2) / A[i][i] 
        if norm_vector([x[i] - x_old[i] for i in range(n)]) < eps:
            return x, k + 1
    return x, max_iter

def main():
    n = 100 
    eps_target = 1e-14 
    
    A_gen = generate_matrix(n)
    x_ideal = [2.5] * n 
    B_gen = generate_B(A_gen, x_ideal)
    
    write_matrix("A.txt", A_gen) 
    write_vector("B.txt", B_gen) 
    
    A = read_matrix("A.txt")
    B = read_vector("B.txt")
    x0 = [1.0] * n 
    
    print(f"Норма матриці A: {norm_matrix(A):.4f}")
    
    methods = [
        ("Проста ітерація", simple_iteration),
        ("Якобі", jacobi),
        ("Зейдель", seidel)
    ]
    
    for name, func in methods:
        sol, iters = func(A, B, x0, eps=eps_target)
        res_norm = get_residual_norm(A, sol, B)
        
        print(f"\n--- {name} ---")
        print(f"Ітерацій: {iters}")
        print(f"Норма нев'язки: {res_norm:.2e}")
        
        filename = f"X_sol_{name.replace(' ', '_')}.txt"
        write_vector(filename, sol)
        print(f"Результат збережено у {filename}")

if __name__ == "__main__":
    main()