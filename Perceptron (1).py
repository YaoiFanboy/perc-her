class Perceptron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.bias = -threshold

    def trigger(self, inputs):
        weighted_sum = sum([w * i for w, i in zip(self.weights, inputs)]) + self.bias
        if weighted_sum > 0:
            return 1
        else:
            return 0

    def __str__(self):
        return f'Perceptron(weights={self.weights}, bias={self.bias})'


# Initialize an AND gate Perceptron
and_gate = Perceptron(weights=[1, 1], threshold=1.5)

# Test the AND gate Perceptron
print('and gate')
print(and_gate.trigger([0, 0]))  # expected output: 0
print(and_gate.trigger([0, 1]))  # expected output: 0
print(and_gate.trigger([1, 0]))  # expected output: 0
print(and_gate.trigger([1, 1]))  # expected output: 1

# Initialize an OR gate Perceptron
or_gate = Perceptron(weights=[1, 1], threshold=0.5)
print('or gate')
# Test the OR gate Perceptron
print(or_gate.trigger([0, 0]))  # expected output: 0
print(or_gate.trigger([0, 1]))  # expected output: 1
print(or_gate.trigger([1, 0]))  # expected output: 1
print(or_gate.trigger([1, 1]))  # expected output: 1

# Initialize a NOR gate Perceptron
nor_gate = Perceptron(weights=[-1, -1, -1], threshold=-1)
print('nor gate')
# Test the NOR gate Perceptron
print(nor_gate.trigger([0, 0, 0]))  # expected output: 1
print(nor_gate.trigger([0, 0, 1]))  # expected output: 0
print(nor_gate.trigger([0, 1, 0]))  # expected output: 0
print(nor_gate.trigger([0, 1, 1]))  # expected output: 0
print(nor_gate.trigger([1, 0, 0]))  # expected output: 0
print(nor_gate.trigger([1, 0, 1]))  # expected output: 0
print(nor_gate.trigger([1, 1, 0]))  # expected output: 0
print(nor_gate.trigger([1, 1, 1]))  # expected output: 0

class PerceptronLayer:
    def __init__(self, perceptrons):
        self.perceptrons = perceptrons
    
    def activate(self, inputs):
        return [p.trigger(inputs) for p in self.perceptrons]
    
    def __str__(self):
        return f"PerceptronLayer(perceptrons={self.perceptrons})"

class PerceptronNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def feed_forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.activate(outputs)
        return outputs
    
    def __str__(self):
        return f"PerceptronNetwork(layers={self.layers})"

# Initialize the XOR gate Perceptron network
xor_network = PerceptronNetwork([
PerceptronLayer([
Perceptron([0.75, -0.75], 0.5), Perceptron([0.5, -1], -1)]),
PerceptronLayer([Perceptron([0.5, -0.5], -0.5)])])

#Test the XOR gate Perceptron network
print('xor')
print(xor_network.feed_forward([0, 0])) # expected output: 0
print(xor_network.feed_forward([0, 1])) # expected output: 1
print(xor_network.feed_forward([1, 0])) # expected output: 1
print(xor_network.feed_forward([1, 1])) # expected output: 0

#Initialize the half adder Perceptron network
half_adder_network = PerceptronNetwork([
    PerceptronLayer([Perceptron([-1, 1], -1), Perceptron([-1, 1], 0.5)]),
    PerceptronLayer([Perceptron([-1, 1], -0.75)]),
    PerceptronLayer([Perceptron([-0.5, -0.5], -0.5), Perceptron([0.5, 0.25], 0.25)])
])


print('half (enige die nog niet klopt)')
#Test the half adder Perceptron network
print(half_adder_network.feed_forward([0, 0])) # expected output: [0, 0]
print(half_adder_network.feed_forward([0, 1])) # expected output: [0, 1]
print(half_adder_network.feed_forward([1, 0])) # expected output: [0, 1]
print(half_adder_network.feed_forward([1, 1])) # expected output: [1, 0]