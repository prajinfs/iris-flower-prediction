from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Get input from user
print("IRIS FLOWER PREDICTION")

sl = float(input("Sepal length: "))
sw = float(input("Sepal width: "))
pl = float(input("Petal length: "))
pw = float(input("Petal width: "))

# Predict flower type
result = model.predict([[sl, sw, pl, pw]])

print("Predicted flower:", iris.target_names[result[0]])