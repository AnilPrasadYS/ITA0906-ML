from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Load built-in Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Perceptron model
model = Perceptron(
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("PERCEPTRON IRIS CLASSIFICATION")
print("------------------------------")

print("Accuracy:",
      round(accuracy * 100, 2), "%")

# New flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

# Predict new flower
prediction = model.predict(new_flower)

print("\nNew Flower Details")
print("Sepal Length:", new_flower[0][0])
print("Sepal Width :", new_flower[0][1])
print("Petal Length:", new_flower[0][2])
print("Petal Width :", new_flower[0][3])

print("\nPredicted Flower:",
      iris.target_names[prediction[0]])
