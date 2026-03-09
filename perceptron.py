import numpy as np


class perceptron:
     
    def __init__(self,input_size):
        
        self.weights = np.random.randn(input_size)
        self.bais = np.random.randn(1)


    def sigmoid(self,Z):
        
        return 1 / (1+np.exp(-Z))

    def predict(self, inputs):

        weighted_sum = np.dot(inputs,self.weights) + self.bais

        return self.sigmoid(weighted_sum)

    def fit(self, inputs, targets, num_epochs, learning_rate):
        num_examples = inputs.shape[0]

        for epoch in range(num_epochs):
            for  i in range(num_examples):
                input_vector = inputs[i]
                target = targets[i]
                prediction = self.predict(input_vector)
                error = target - prediction
                gradient_weights = error * prediction * (1 - prediction)  
                self.weights += learning_rate * gradient_weights
                gradient_bias = error*prediction*(1-prediction)
                self.bias += learning_rate* gradient_bias
            
        print(f"Epoch {epoch+1}/{num_epochs} done!")

    def evaluate(self,inputs,target):
        correct = 0
        for input_vector,target in zip(input,target):
            prediction = self.predict(input_vector)
            if prediction >=0.5:
                prediction_class=1

            else:
                prediction_class = 0

            if prediction_class == target:
                correct+=1


        accuracy = correct / len (inputs)

        return accuracy
    