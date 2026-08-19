import numpy as np
import matplotlib.pyplot as plt
hours = np.array([1, 2, 3, 4, 5, 6])
marks = np.array([40, 50, 55, 65, 70, 85])
plt.scatter(hours, marks)
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()