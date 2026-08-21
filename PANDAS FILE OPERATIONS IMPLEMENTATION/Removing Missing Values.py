import pandas as pd
import numpy as np
data = {
    "Name": ["Anil", "Ravi", "Priya", "Kiran"],
    "Marks": [85, np.nan, 90, np.nan],
    "Age": [20, 21, np.nan, 22]
}
df = pd.DataFrame(data)
print("Original Data:")
print(df)
result = df.dropna()
print("\nAfter removing missing values:")
print(result)