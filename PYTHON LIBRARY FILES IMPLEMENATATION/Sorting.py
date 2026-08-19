import numpy as np
prices = np.array([500, 100, 300, 50, 200])
print("Original:", prices)
print("Ascending:", np.sort(prices))
print("Descending:", np.sort(prices)[::-1])