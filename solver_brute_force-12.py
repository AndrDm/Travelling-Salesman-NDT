import pandas as pd
import numpy as np
import itertools

print("Brute Force Solver")
# Load the CSV file
df = pd.read_csv("cost_matrix_12.csv", header=None, delimiter=";")
cost_matrix = df.to_numpy(dtype=float)

# Number of Positions
n = len(cost_matrix)
# Generate all possible permutations of positions (excluding the starting pos)
positions = list(range(1, n))
min_time = float('inf')
optimal_inspection = []

# Brute force search for the optimal inspection
for perm in itertools.permutations(positions):
    inspection = [0] + list(perm) + [0]  # start and end at position 0
    time = sum(cost_matrix[inspection[i]][inspection[i+1]] for i in range(n))
    if time < min_time:
        min_time = time
        optimal_inspection = inspection
        print(f"Sequence: {inspection} | Time: {time:.3f} s")

# Print the optimal inspection and its total cost
print("Optimal Sequence:", optimal_inspection)
print("Total Time, s:", min_time)

