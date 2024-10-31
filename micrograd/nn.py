import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

    def __str__(self):
        fstr = f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)}) -> "
        for idx, wi in enumerate(self.w):
            fstr += f"w{idx} ={wi.data:7.4f}, "
        fstr += f"b ={self.b.data:7.4f}"
        return fstr
class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n.__repr__()) for n in self.neurons)}]"

    def __str__(self):
        fstr = f"Shape of the layer is: {len(self.neurons[0].w)} X {len(self.neurons)} (nin X nout)\n"
        for idx, neuron in enumerate(self.neurons):
            fstr += f"   Neuron {idx+1}: {neuron.__str__()} \n"
        return fstr

# Neurons are non-linear in every layer except the last one.
class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
         return f"MLP of [{', '.join(str(layer.__repr__()) for layer in self.layers)}]"

    def __str__(self):
        fstr = "Multi-Layer Perceptron Structure:\n"
        for idx, layer in enumerate(self.layers):
            fstr += f" Layer {idx+1}/{len(self.layers)} - {layer}\n"
        return fstr