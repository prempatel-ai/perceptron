import numpy as np


class Perceptron:

    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.loss_history = []

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def predict(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(z)

    def fit(self, inputs, targets, num_epochs, learning_rate):
        num_examples = inputs.shape[0]
        self.loss_history = []

        for epoch in range(num_epochs):
            epoch_loss = 0

            for i in range(num_examples):
                input_vector = inputs[i]
                target = targets[i]
                prediction = self.predict(input_vector)
                error = target - prediction

                epoch_loss += error ** 2

                gradient_weights = error * prediction * (1 - prediction) * input_vector
                self.weights += learning_rate * gradient_weights

                gradient_bias = error * prediction * (1 - prediction)
                self.bias += learning_rate * gradient_bias

            avg_loss = epoch_loss / num_examples
            self.loss_history.append(float(avg_loss))
            print(f"Epoch {epoch+1}/{num_epochs}  |  Loss: {round(float(avg_loss), 4)}")

    def evaluate(self, inputs, targets):
        correct = 0
        for input_vector, target in zip(inputs, targets):
            prediction = self.predict(input_vector)
            predicted_class = 1 if prediction >= 0.5 else 0
            if predicted_class == target:
                correct += 1
        return correct / len(inputs)

    def save(self, filename):
        np.save(filename, {'weights': self.weights, 'bias': self.bias})
        print(f"saved to {filename}.npy")

    def load(self, filename):
        data = np.load(filename, allow_pickle=True).item()
        self.weights = data['weights']
        self.bias = data['bias']
        print(f"loaded from {filename}.npy")