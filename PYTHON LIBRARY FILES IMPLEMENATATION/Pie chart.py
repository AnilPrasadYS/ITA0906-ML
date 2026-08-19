import numpy as np
import matplotlib.pyplot as plt
items = np.array(["Food", "Travel", "Books", "Others"])
expenses = np.array([40, 25, 20, 15])
plt.pie(expenses, labels=items, autopct="%1.1f%%")
plt.title("Monthly Expenses")
plt.show()