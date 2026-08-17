import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create car dataset
data = {
    "Age": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Mileage": [10000, 20000, 30000, 40000, 50000,
                60000, 70000, 80000, 90000, 100000],
    "Engine": [1500, 1500, 1600, 1600, 1800,
               1800, 2000, 2000, 2200, 2200],
    "Price": [900000, 820000, 750000, 680000, 600000,
              530000, 470000, 400000, 350000, 300000]
}

df = pd.DataFrame(data)

# Input features
X = df[["Age", "Mileage", "Engine"]]

# Target
y = df["Price"]

# Split data
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

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Car Price Prediction")
print("--------------------")

print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))

# New car
new_car = pd.DataFrame({
    "Age": [3],
    "Mileage": [30000],
    "Engine": [1600]
})

# Predict price
prediction = model.predict(new_car)

print("\nNew Car Details")
print("Age:", new_car["Age"].iloc[0], "years")
print("Mileage:", new_car["Mileage"].iloc[0], "km")
print("Engine:", new_car["Engine"].iloc[0], "cc")

print("\nPredicted Car Price: ₹",
      round(prediction[0], 2))
