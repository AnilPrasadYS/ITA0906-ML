import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# House dataset
# [Area, Bedrooms, Bathrooms, Age]
X = np.array([
    [1000, 2, 1, 15],
    [1200, 2, 2, 12],
    [1400, 3, 2, 10],
    [1600, 3, 2, 8],
    [1800, 3, 3, 7],
    [2000, 4, 3, 6],
    [2200, 4, 3, 5],
    [2400, 4, 4, 4],
    [2600, 5, 4, 3],
    [2800, 5, 4, 2]
])

# House prices
y = np.array([
    3000000,
    3500000,
    4200000,
    4800000,
    5500000,
    6200000,
    6800000,
    7500000,
    8300000,
    9000000
])

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("HOUSE PRICE PREDICTION")
print("----------------------")

print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))

# New house
new_house = np.array([[1700, 3, 2, 5]])

# Predict price
prediction = model.predict(new_house)

print("\nNew House Details")
print("Area:", new_house[0][0], "sq.ft")
print("Bedrooms:", new_house[0][1])
print("Bathrooms:", new_house[0][2])
print("Age:", new_house[0][3], "years")

print("\nPredicted House Price: ₹",
      round(prediction[0], 2))
