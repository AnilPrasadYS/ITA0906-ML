import numpy as np
import matplotlib.pyplot as plt
marks = np.array([40, 45, 50, 55, 60, 60, 65, 70, 70, 70, 75, 80, 85, 90, 90])
plt.hist(marks, bins=5, edgecolor="black")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.title("Marks Distribution")
plt.show()