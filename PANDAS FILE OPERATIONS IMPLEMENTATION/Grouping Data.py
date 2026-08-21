import pandas as pd
data = {
    "Name": ["Anil", "Ravi", "Priya", "Kiran", "Sneha"],
    "Department": ["ECE", "CSE", "ECE", "CSE", "ECE"],
    "Marks": [85, 70, 90, 65, 80]
}
df = pd.DataFrame(data)
print("Original Data:")
print(df)
result = df.groupby("Department")["Marks"].mean()
print("\nAverage marks by department:")
print(result)