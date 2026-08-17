import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Mobile dataset
# [RAM(GB), Storage(GB), Battery(mAh), Camera(MP), Screen(inches)]

X = np.array([
    [2, 32, 3000, 12, 5.0],
    [3, 32, 3500, 13, 5.2],
    [4, 64, 4000, 16, 5.5],
    [4, 128, 4500, 20, 6.0],
    [6, 128, 4500, 32, 6.1],
    [6, 256, 5000, 48, 6.4],
    [8, 128, 5000, 50, 6.5],
    [8, 256, 5500, 64, 6.6],
    [12, 256, 5000, 108, 6.7],
    [12, 512, 6000, 108, 6.8]
])

# Mobile prices
y = np.array([
    10000,
    12000,
    15000,
    18000,
    22000,
    28000,
    32000,
    38000,
    48000,
    55000
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
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

print("MOBILE PRICE PREDICTION")
print("-----------------------")

print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))

# New mobile
new_mobile = np.array([
    [8, 256, 5000, 64, 6.5]
])

# Predict price
prediction = model.predict(new_mobile)

print("\nNew Mobile Specifications")
print("RAM:", new_mobile[0][0], "GB")
print("Storage:", new_mobile[0][1], "GB")
print("Battery:", new_mobile[0][2], "mAh")
print("Camera:", new_mobile[0][3], "MP")
print("Screen Size:", new_mobile[0][4], "inches")

print("\nPredicted Mobile Price: ₹",
      round(prediction[0], 2))
