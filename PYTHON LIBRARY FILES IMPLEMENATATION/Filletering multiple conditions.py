import numpy as np
age = np.array([20, 21, 20, 23, 21])
marks = np.array([85, 65, 92, 55, 78])
result = marks[(marks > 70) & (age < 22)]
print("Filtered marks:", result)