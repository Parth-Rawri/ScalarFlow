import numpy as np
from autograd.engine import Value

class Neuron:
    """
    nin: number of inputs to the neuron
    """
    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1,1))

    def __call__(self, x):
        preact = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return preact.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    """
    nin: number of inputs each neuron in the layer expects
    nout: number of outputs/neurons
    """
    def __init__(self, nin, nout): 
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    """
    nin: number of inputs/features to network
    nouts: number of outputs/neurons for each layer 
        eg: If nouts = [4, 3, 2], means layer1 has 4 neurons, layer2 has 3 neurons, and layer3 (output layer) has 2 neurons
    sz: defines the connectivity between the input layer and the subsequent layers 
        eg: If sz = [3, 4, 3, 2], where 3 represent the number of inputs to network
    """
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

if __name__ == "__main__":
    xs = [
        [2.0, 3.0, -1.0],
        [4.0, -2.0, 2.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, 2.0],
        ]
    ys = [1.0, 1.0, -1.0, 1.0] 
    model = MLP(nin=3, nouts=[4, 4, 1])

    for k in range(1000):
        # forward pass
        ypred = [model(x) for x in xs]
        loss = sum((ypred - ygt)**2 for ygt, ypred in zip(ys, ypred))
        
        # backward pass
        for param in model.parameters():
            param.grad = 0.0
        loss.backward()
        
        # update all the parameters of the Neural network
        for param in model.parameters():
            param.data += -0.01 * param.grad
        print(k, loss.data)

    print(ypred)