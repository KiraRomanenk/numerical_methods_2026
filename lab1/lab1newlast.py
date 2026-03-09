import requests
import numpy as np
import matplotlib.pyplot as plt

def tridiagonal_matrix_algorithm(h, y):
    n = len(h) + 1
    alpha = np.zeros(n)
    beta = np.zeros(n)
    F = np.zeros(n)
    
    for i in range(1, n-1):
        diff1 = (y[i+1] - y[i]) / h[i]
        diff2 = (y[i] - y[i-1]) / h[i-1]
        F[i] = 3 * (diff1 - diff2)
    alpha[1] = 0
    beta[1] = 0
    
    for i in range(1, n-1):
        A_i = h[i-1]
        B_i = 2 * (h[i-1] + h[i])
        C_i = h[i]
        
        denom = B_i + A_i * alpha[i]
        alpha[i+1] = -C_i / denom
        beta[i+1] = (F[i] - A_i * beta[i]) / denom

    c = np.zeros(n)
    c[-1] = 0 
    
    for i in range(n-2, 0, -1):
        c[i] = alpha[i+1] * c[i+1] + beta[i+1]
        
    return c

def calculate_spline_coefficients(x, y, c):
    n = len(x) - 1
    h = np.diff(x)
    
    a = np.zeros(n)
    b = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(n):
        a[i] = y[i] 
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
        b[i] = (y[i+1] - y[i]) / h[i] - (h[i] / 3) * (c[i+1] + 2 * c[i])
        
    return a, b, c[:-1], d

def interpolate(x_query, x_nodes, a, b, c, d):
    if x_query < x_nodes[0] or x_query > x_nodes[-1]:
        return None

    i = 0
    for k in range(len(x_nodes) - 1):
        if x_nodes[k] <= x_query <= x_nodes[k+1]:
            i = k
            break
            
    dx = x_query - x_nodes[i]
    val = a[i] + b[i]*dx + c[i]*(dx**2) + d[i]*(dx**3)
    return val

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def get_data():
    # Coordinates from PDF (Zaroslyak - Hoverla)
    coords_str = [
        (48.164214, 24.536044), (48.164983, 24.534836), (48.165605, 24.534068),
        (48.166228, 24.532915), (48.166777, 24.531927), (48.167326, 24.530884),
        (48.167011, 24.530061), (48.166053, 24.528039), (48.166655, 24.526064),
        (48.166497, 24.523574), (48.166128, 24.520214), (48.165416, 24.517170),
        (48.164546, 24.514640), (48.163412, 24.512980), (48.162331, 24.511715),
        (48.162015, 24.509462), (48.162147, 24.506932), (48.161751, 24.504244),
        (48.161197, 24.501793), (48.160580, 24.500537), (48.160250, 24.500106)
    ]

    loc_str = "|".join([f"{lat},{lon}" for lat, lon in coords_str])
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={loc_str}"
    
    print("Requesting API Open-Elevation...")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            results = response.json()['results']
        else:
            print("API Error, using fallback data.")
            raise Exception("API Error")
    except Exception as e:
        print(f"Failed to get data ({e}). Generating test elevations.")
        results = []
        base_h = 1200
        for i, (lat, lon) in enumerate(coords_str):
            results.append({'latitude': lat, 'longitude': lon, 'elevation': base_h + i*40})

    return results

def main():
    results = get_data()
    
    n_points = len(results)
    coords = [(p['latitude'], p['longitude']) for p in results]
    elevations = np.array([p['elevation'] for p in results])
    distances = [0.0]
    for i in range(1, n_points):
        d = haversine(*coords[i-1], *coords[i])
        distances.append(distances[-1] + d)
    
    distances = np.array(distances)
    
    print(f"\nNode count: {n_points}")
    print("-" * 50)
    print(f"{'No':<4} | {'Dist (m)':<10} | {'Elev (m)':<10} | {'Lat':<10} | {'Lon':<10}")
    print("-" * 50)
    for i in range(n_points):
        print(f"{i:<4d} | {distances[i]:<10.2f} | {elevations[i]:<10.2f} | {coords[i][0]:<10.6f} | {coords[i][1]:<10.6f}")
    
    with open("route_data.txt", "w") as f:
        f.write("Index, Distance, Elevation\n")
        for i in range(n_points):
            f.write(f"{i}, {distances[i]}, {elevations[i]}\n")

    h = np.diff(distances)
    
    c_coeffs_full = tridiagonal_matrix_algorithm(h, elevations)
    a, b, c, d = calculate_spline_coefficients(distances, elevations, c_coeffs_full)
    
    print("\nSpline Coefficients (first 5):")
    print(f"{'i':<3} | {'a':<10} | {'b':<10} | {'c':<10} | {'d':<10}")
    for i in range(min(5, len(a))):
        print(f"{i:<3} | {a[i]:<10.4f} | {b[i]:<10.4f} | {c[i]:<10.4f} | {d[i]:<10.6f}")
    
    x_dense = np.linspace(distances[0], distances[-1], 500)
    y_dense = [interpolate(x, distances, a, b, c, d) for x in x_dense]
    
    plt.figure(figsize=(12, 6))
    plt.plot(distances, elevations, 'ro', label='GPS Nodes (Original)')
    plt.plot(x_dense, y_dense, 'b-', label='Cubic Spline')
    plt.title("Elevation Profile: Zaroslyak - Hoverla")
    plt.xlabel("Distance (m)")
    plt.ylabel("Elevation (m)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_dense, y_dense, 'k--', alpha=0.3, label='Full Spline (Reference)')
    
    subsets = [10, 15] 
    colors = ['g', 'orange']
    errors = [] 
    
    for idx, num_nodes in enumerate(subsets):
        indices = np.linspace(0, n_points - 1, num_nodes, dtype=int)
        
        sub_dist = distances[indices]
        sub_elev = elevations[indices]
        
        sub_h = np.diff(sub_dist)
        sub_c_full = tridiagonal_matrix_algorithm(sub_h, sub_elev)
        sa, sb, sc, sd = calculate_spline_coefficients(sub_dist, sub_elev, sub_c_full)
        
        y_sub_dense = [interpolate(x, sub_dist, sa, sb, sc, sd) for x in x_dense]
        
        current_error = np.abs(np.array(y_dense) - np.array(y_sub_dense))
        errors.append(current_error)
        
        plt.plot(x_dense, y_sub_dense, color=colors[idx], label=f'Spline ({num_nodes} nodes)')
        plt.scatter(sub_dist, sub_elev, color=colors[idx], s=20)

    plt.title("Impact of Node Count on Accuracy")
    plt.xlabel("Distance (m)")
    plt.ylabel("Elevation (m)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    for idx, num_nodes in enumerate(subsets):
        plt.plot(x_dense, errors[idx], color=colors[idx], label=f'Error ({num_nodes} nodes)')
        
    plt.title("Interpolation Error (ε = |y - y_approx|)")
    plt.xlabel("Distance (m)")
    plt.ylabel("Absolute Error (m)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n--- Additional Analysis ---")
    
    total_length = distances[-1]
    total_ascent = 0
    total_descent = 0
    
    for i in range(1, len(elevations)):
        diff = elevations[i] - elevations[i-1]
        if diff > 0:
            total_ascent += diff
        else:
            total_descent += abs(diff)
            
    print(f"Total length: {total_length:.2f} m")
    print(f"Total ascent: {total_ascent:.2f} m")
    print(f"Total descent: {total_descent:.2f} m")
    
    gradients = [] # in %
    for x in x_dense:
        k = 0
        for j in range(len(distances)-1):
            if distances[j] <= x <= distances[j+1]:
                k = j
                break
        dx = x - distances[k]
        slope = b[k] + 2*c[k]*dx + 3*d[k]*(dx**2)
        gradients.append(slope * 100)
        
    gradients = np.array(gradients)
    print(f"Max ascent: {np.max(gradients):.2f}%")
    print(f"Max descent: {np.min(gradients):.2f}%")
    print(f"Average gradient (abs): {np.mean(np.abs(gradients)):.2f}%")
    
    # 3. Mechanical Energy (mass 80kg)
    mass = 80
    g = 9.81
    energy_joules = mass * g * total_ascent
    energy_kcal = energy_joules / 4184
    
    print(f"Mechanical work: {energy_joules/1000:.2f} kJ")
    print(f"Energy expenditure (approx): {energy_kcal:.2f} kcal")

if __name__ == "__main__":
    main()