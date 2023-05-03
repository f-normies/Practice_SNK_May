import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        
        #Веса между входным и скрытым слоями, скрытым и выходным слоями
        self.weights_ih = np.random.randn(hidden_nodes, input_nodes) * 0.01
        self.weights_ho = np.random.randn(output_nodes, hidden_nodes) * 0.01
        
        #Вводим пороги для скрытого и выходного слоев
        self.bias_h = np.zeros((hidden_nodes, 1))
        self.bias_o = np.zeros((output_nodes, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        #np.dot - векторное произведение
        #хотим получить значения после применения сигмоиды (значения активации)
        self.z_h = np.dot(self.weights_ih, X) + self.bias_h
        self.a_h = self.sigmoid(self.z_h)
        self.z_o = np.dot(self.weights_ho, self.a_h) + self.bias_o
        self.a_o = self.sigmoid(self.z_o)
        return self.a_o

    def train(self, X, y):
        # Forward propagation
        output = self.forward(X)
        
        # Ошибка выходного слоя
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        # Ошибка скрытого слоя
        hidden_error = np.dot(self.weights_ho.T, output_delta)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a_h)

        # Обновляем ошибки и веса
        self.weights_ho += self.learning_rate * np.dot(output_delta, self.a_h.T)
        self.weights_ih += self.learning_rate * np.dot(hidden_delta, X.T)
        self.bias_o += self.learning_rate * np.sum(output_delta, axis=1, keepdims=True)
        self.bias_h += self.learning_rate * np.sum(hidden_delta, axis=1, keepdims=True)

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

# Входные данные
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Нормализуем данные
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# One-hot энкодинг
y_encoded = np.zeros((y.size, y.max() + 1))
y_encoded[np.arange(y.size), y] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Создаем объект класса MLP с параметрами
mlp = MLP(input_nodes=4, hidden_nodes=10, output_nodes=3, learning_rate=0.1)

# Тренируем модель
epochs = 1000
loss_history = []

for epoch in range(epochs):
    mlp.train(X_train.T, y_train.T)
    predictions = mlp.forward(X_train.T)
    loss = np.mean(-y_train * np.log(predictions.T) - (1 - y_train) * np.log(1 - predictions.T))
    loss_history.append(loss)
    
# Оцениваем модель
test_predictions = mlp.forward(X_test.T)
test_accuracy = accuracy(y_test, test_predictions.T)
print(f"Test accuracy: {test_accuracy * 100}%")
print(f"Loss: {loss_history[-1]}")

# Графики ошибки
plt.plot(loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss history')
plt.show()