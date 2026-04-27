from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train models
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

# Accuracy
rf_acc = accuracy_score(y_test, rf.predict(X_test))
lr_acc = accuracy_score(y_test, lr.predict(X_test))

print("Random Forest:", rf_acc)
print("Logistic Regression:", lr_acc)

# Save model
joblib.dump(rf, "model.pkl")