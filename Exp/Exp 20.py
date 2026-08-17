import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Previous monthly sales data
# Month number
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

# Sales
y = np.array([
    100,
    120,
    135,
    150,
    165,
    180,
    200,
    215,
    230,
    250
])

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict existing sales
y_pred = model.predict(X)

# Calculate performance
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("FUTURE SALES PREDICTION")
print("-----------------------")

print("Mean Squared Error:",
      round(mse, 2))

print("R2 Score:",
      round(r2, 2))

# Predict future months
future_months = np.array([
    [11],
    [12],
    [13]
])

future_sales = model.predict(future_months)

print("\nFuture Sales Prediction")

for i in range(len(future_months)):
    print("Month", future_months[i][0],
          ":", round(future_sales[i], 2), "units")
