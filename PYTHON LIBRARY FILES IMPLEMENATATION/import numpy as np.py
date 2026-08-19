import numpy as np
data = np.array([
    [1, 20, 85],
    [2, 21, 65],
    [3, 20, 92],
    [4, 22, 55],
    [5, 21, 78]
])
print("Original Data:")
print(data)
filtered_data = data[data[:, 2] > 75]
print("\nFiltered Data:")
print(filtered_data)