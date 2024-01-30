import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ArtificialNeuron:
    def __init__(self, num_inputs): # Initialize weight and bias
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()

    def forward(self, inputs): # Calculate the weighted sum of the inputs and apply the sigmoid activation function
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = sigmoid(weighted_sum)
        return output

# Example
num_inputs = 2
neuron = ArtificialNeuron(num_inputs)
# Generate random data points for visualization
num_points = 100
random_inputs = np.random.rand(num_points, num_inputs)
# calculate the output for each points
outputs = [neuron.forward(point) for point in random_inputs]
# scatter plot with color representing the neuron's output
plt.scatter(random_inputs[:, 0], random_inputs[:, 1], c=outputs, cmap='viridis')
plt.title('Decision Boundry Of Artificial Neurons')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.colorbar(label='Neuron Output')
plt.show()





