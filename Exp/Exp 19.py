import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Bank loan dataset
# [Income, Loan Amount, Credit Score, Employment Years]

X = np.array([
    [25000, 100000, 550, 2],
    [30000, 120000, 580, 3],
    [35000, 150000, 600, 4],
    [40000, 180000, 620, 5],
    [45000, 200000, 650, 6],
    [50000, 220000, 680, 7],
    [60000, 250000, 700, 8],
    [70000, 300000, 720, 9],
    [80000, 350000, 750, 10],
    [90000, 400000, 780, 12]
])

# Loan status
# 0 = Rejected
# 1 = Approved

y = np.array([
    0, 0, 0, 0, 1,
    1, 1, 1, 1, 1
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Naive Bayes model
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("BANK LOAN PREDICTION")
print("-------------------")

print("Accuracy:",
      round(accuracy * 100, 2), "%")

# New customer
new_customer = np.array([
    [55000, 230000, 690, 7]
])

# Predict loan status
prediction = model.predict(new_customer)

print("\nNew Customer Details")
print("Income:", new_customer[0][0])
print("Loan Amount:", new_customer[0][1])
print("Credit Score:", new_customer[0][2])
print("Employment Years:", new_customer[0][3])

if prediction[0] == 1:
    print("\nLoan Status: APPROVED")
else:
    print("\nLoan Status: REJECTED")
