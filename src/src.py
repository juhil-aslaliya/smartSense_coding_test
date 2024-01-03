import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class LogisticRegression:
    def __init__(self, lr=0.01, iter=1000):
        self.lr = lr
        self.iter = iter
        self.models = {}

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def init_params(self, num_features):
        return np.zeros(num_features)

    def cost(self, X, y, theta):
        m = len(y)
        pred = self.sigmoid(np.dot(X, theta))
        cost = (-1/m) * np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        return cost

    def gradient_descent(self, X, y, theta):
        m = len(y)
        pred = self.sigmoid(np.dot(X, theta))
        grad = np.dot(X.T, (pred - y)) / m
        return grad

    def train_1va(self, X, y, label):
        theta = self.init_params(X.shape[1])
        # Training for one label
        for i in range(self.iter):
            cost = self.cost(X, y, theta)
            grad = self.gradient_descent(X, y, theta)
            theta -= self.lr * grad
            if i % (self.iter//10) == 0:
                print(f"Iteration {i}, Label {label}, Cost: {cost}")
        return theta

    def train(self, X_train, y_train):
        unique_labels = np.arange(y_train.shape[1])
        # Training for each label
        for label in unique_labels:
            print("--------------------------------------------------------------------------------------------------------------------------")
            binary_model = self.train_1va(X_train, y_train[:, label], label)
            self.models[label] = binary_model

    def predict_1va(self, X, target_label):
        # Predicting for one label
        scores = np.dot(X, self.models[target_label])
        return self.sigmoid(scores)

    def predict(self, X):
        # Predicting for all labels
        pred = np.array([self.predict_1va(X, label)
                        for label in self.models.keys()]).T
        return pred


# Loading the data
traindata = pd.read_csv("../data/train.csv")
testdata = pd.read_csv("../data/test.csv")
features = ["Length", "Width", "Height", "Wheelbase"]
targets = ["SUV", "MUV", "Sedan", "Hatchback"]
X_train = traindata[features].values
y_train = traindata[targets].values
X_test = testdata[features].values
y_test = testdata[targets].values

# Normalizing the features
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Adding the bias term
X_train_norm = np.c_[np.ones(X_train_norm.shape[0]), X_train_norm]
X_test_norm = np.c_[np.ones(X_test_norm.shape[0]), X_test_norm]

# Training
print("--------------------------------------------------------------------------------------------------------------------------")
print("Training")
# lr and iter can be hyperparameters here
logreg = LogisticRegression(lr=1, iter=10000)
logreg.train(X_train_norm, y_train)

# Prediction and results
pred = logreg.predict(X_test_norm)
acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))
print("--------------------------------------------------------------------------------------------------------------------------")
print(f"Accuracy: {acc}")
print("--------------------------------------------------------------------------------------------------------------------------")
print("Predicted values: ")
for i in range(pred.shape[0]):
    print(f"{pred[i]}, Pedicted Label: {targets[np.argmax(pred[i])]}, Real Label: {targets[np.argmax(y_test[i])]}")
print("--------------------------------------------------------------------------------------------------------------------------")
print("Models: ")
for i in logreg.models:
    print(f"{targets[i]}: {logreg.models[i]}")
print("--------------------------------------------------------------------------------------------------------------------------")
