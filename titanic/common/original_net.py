import sys
sys.path.append('..')
import numpy as np
from common.layers import Affine, Sigmoid, ReLU, SoftmaxWithLoss
from common.base_model import BaseModel

class OriginalNet(BaseModel):
    def __init__(self, input_size, hidden_sizes, output_size):
        I, O = input_size, output_size
        previous_H = I
        current_H = hidden_sizes[0]
        h_length = len(hidden_sizes)
        self.layers = []
        for i in range(h_length):
            current_H = hidden_sizes[i]
            W = 0.01 * np.random.randn(previous_H, current_H)
            b = np.zeros(current_H)
            self.layers.append(Affine(W, b))
            self.layers.append(Sigmoid())
            #self.layers.append(ReLU())
            previous_H = current_H
        W = 0.01 * np.random.randn(previous_H, O)
        b = np.zeros(O)
        self.layers.append(Affine(W, b))

        self.loss_layer = SoftmaxWithLoss()

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def show(self):
        for layer in self.layers:
            print(layer.params)

