import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode the target labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the multi-layer perceptron with multiple hidden layers
class MLP:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
            self.biases.append(np.zeros(layer_sizes[i+1]))

    def forward(self, X):
        self.a = [X]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.a[-1], w) + b
            self.a.append(sigmoid(z))
        return self.a[-1]

    def backward(self, X, y, output, learning_rate):
        deltas = [output - y]
        for i in range(len(self.weights)-1, 0, -1):
            deltas.append(np.dot(deltas[-1], self.weights[i].T) * sigmoid_derivative(self.a[i]))

        deltas.reverse()
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.a[i].T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0)

    def train(self, X, y, learning_rate, epochs):
        loss_history = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            loss = np.mean(np.square(y - output))
            loss_history.append(loss)
        return loss_history

# Initialize and train the multi-layer perceptron
layer_sizes = [4, 8, 8, 3]  # 4 input nodes, two hidden layers with 8 nodes each, and 3 output nodes
mlp = MLP(layer_sizes)
loss_history = mlp.train(X_train, y_train, learning_rate=0.01, epochs=150)

# Evaluate the trained model
predictions = np.argmax(mlp.forward(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(predictions, y_test_labels)
print(f"Accuracy: {accuracy:.2f}")

# Plot the loss history
plt.plot(loss_history)
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()