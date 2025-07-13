import pandas as pd
import numpy as np

print("Greedy Solver")
# Load the CSV file
df = pd.read_csv("cost_matrix_60.csv", header=None, delimiter=";")
cost_matrix = df.to_numpy(dtype=float)

# Greedy Inspection optimizer
def solve_tsp_greedy(matrix):
    n = len(matrix)
    inspected = [False] * n
    path = [0]
    inspected[0] = True
    total_cost = 0

    for _ in range(n - 1):
        last = path[-1]
        next_pos = np.argmin([matrix[last][j] if not inspected[j] else np.inf for j in range(n)])
        path.append(next_pos)
        inspected[next_pos] = True
        total_cost += matrix[last][next_pos]

    total_cost += matrix[path[-1]][path[0]]
    path.append(path[0])
    route = [int(x) for x in path]
    return route, total_cost

# Solve the TSP
insp_route, insp_cost = solve_tsp_greedy(cost_matrix)

# Output results
print("Optimal Inspection Path:", insp_route)
print("Total Time, s:", round(insp_cost, 3))
