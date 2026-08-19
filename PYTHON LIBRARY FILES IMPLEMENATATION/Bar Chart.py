import numpy as np
import matplotlib.pyplot as plt
students = np.array(["Anil", "Ravi", "Priya", "Kiran"])
marks = np.array([85, 70, 95, 60])
plt.bar(students, marks)
plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Student Marks")
plt.show()