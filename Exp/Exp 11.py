import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Create dataset
data = {
    "Income": [25000, 30000, 45000, 50000, 60000,
               70000, 80000, 90000, 35000, 40000],

    "LoanAmount": [20000, 25000, 30000, 25000, 20000,
                   15000, 10000, 12000, 30000, 28000],

    "PaymentHistory": [1, 1, 0, 1, 1,
                       1, 1, 1, 0, 0],

    "CreditHistory": [1, 1, 0, 1, 1,
                      1, 1, 1, 0, 0],

    "CreditScore": [
        "Poor", "Average", "Poor", "Good", "Good",
        "Good", "Good", "Good", "Poor", "Average"
    ]
}

df = pd.DataFrame(data)

# Input features
X = df[[
    "Income",
    "LoanAmount",
    "PaymentHistory",
    "CreditHistory"
]]

# Target
y = df["CreditScore"]

# Encode target
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Create model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X, y_encoded)

# New customer as DataFrame
new_customer = pd.DataFrame({
    "Income": [55000],
    "LoanAmount": [22000],
    "PaymentHistory": [1],
    "CreditHistory": [1]
})

# Prediction
prediction = model.predict(new_customer)

# Convert back to text
result = encoder.inverse_transform(prediction)

print("Credit Score Classification")
print("---------------------------")

print("Income:", new_customer["Income"].iloc[0])
print("Loan Amount:", new_customer["LoanAmount"].iloc[0])
print("Payment History:", new_customer["PaymentHistory"].iloc[0])
print("Credit History:", new_customer["CreditHistory"].iloc[0])

print("\nPredicted Credit Score:", result[0])
