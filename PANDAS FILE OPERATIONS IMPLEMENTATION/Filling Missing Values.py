import pandas as pd
import numpy as np
data = {
    "Name": ["Anil", "Ravi", "Priya", "Kiran"],
    "Marks": [85, np.nan, 90, np.nan]
}
df = pd.DataFrame(data)
print("Original Data:")
print(df)
average = df["Marks"].mean()
df["Marks"] = df["Marks"].fillna(average)
print("\nAfter filling missing values:")
print(df)