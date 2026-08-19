import numpy as np
import matplotlib.pyplot as plt

days = np.array([1, 2, 3, 4, 5])
temperature = np.array([30, 32, 31, 35, 33])
plt.plot(days, temperature, marker='o')
plt.xlabel("Days")
plt.ylabel("Temperature")
plt.title("Temperature for 5 Days")
plt.show()