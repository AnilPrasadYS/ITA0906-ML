from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load Built-in Wine Dataset
wine = load_wine()

# Features and Target
X = wine.data
y = wine.target

# Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Create Naive Bayes Model
model = GaussianNB()

# Train the Model
model.fit(X_train, y_train)

# Predict Test Data
y_pred = model.predict(X_test)

# Display Actual Labels
print("Actual Labels:")
print(y_test)

# Display Predicted Labels
print("\nPredicted Labels:")
print(y_pred)

# Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Display Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy = {:.2f}%".format(accuracy * 100))
