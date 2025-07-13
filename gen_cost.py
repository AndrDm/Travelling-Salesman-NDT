import numpy as np

def generate_cost_matrix(size, low=1.0, high=2.0, filename="cost_matrix.csv"):
    mat = np.random.uniform(low, high, (size, size)) # Generate random floats
    np.fill_diagonal(mat, 0.0) # Set diagonal to zero
    mat_str = np.vectorize(lambda x: f"{x:.3f}")(mat) # Format each value
    with open(filename, "w") as f:
        for row in mat_str:
            f.write(";".join(row) + "\n") # Write to file with semicolon sep.

# Generate for 12 and 60 positions
generate_cost_matrix(12, filename="cost_matrix_12.csv")
generate_cost_matrix(60, filename="cost_matrix_60.csv")
