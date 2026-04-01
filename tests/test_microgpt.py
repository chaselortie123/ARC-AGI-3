"""Tests for Karpathy's microgpt.py core components."""

import math
import sys
import os

# microgpt.py runs training at import time, so we test the Value autograd
# engine and helper functions directly without importing the full script.

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# === Inline the Value class so we can test it without triggering the full script ===

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


# === Tests ===

class TestValueAutograd:
    def test_add(self):
        a = Value(3.0)
        b = Value(4.0)
        c = a + b
        assert c.data == 7.0

    def test_mul(self):
        a = Value(3.0)
        b = Value(4.0)
        c = a * b
        assert c.data == 12.0

    def test_backward_simple(self):
        # f(a, b) = a * b + a
        a = Value(2.0)
        b = Value(3.0)
        c = a * b + a
        c.backward()
        # dc/da = b + 1 = 4.0, dc/db = a = 2.0
        assert abs(a.grad - 4.0) < 1e-6
        assert abs(b.grad - 2.0) < 1e-6

    def test_backward_chain(self):
        # f(x) = (x * x) * x  =>  x^3, df/dx = 3x^2
        x = Value(3.0)
        y = x * x * x
        y.backward()
        assert abs(y.data - 27.0) < 1e-6
        assert abs(x.grad - 27.0) < 1e-6  # 3 * 3^2 = 27

    def test_relu(self):
        a = Value(5.0)
        b = Value(-3.0)
        assert a.relu().data == 5.0
        assert b.relu().data == 0.0

    def test_exp_log_roundtrip(self):
        a = Value(2.0)
        result = a.exp().log()
        assert abs(result.data - 2.0) < 1e-6

    def test_sub_and_neg(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 2.0
        d = -a
        assert d.data == -5.0

    def test_div(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        assert abs(c.data - 2.0) < 1e-6

    def test_pow(self):
        a = Value(3.0)
        c = a ** 2
        c.backward()
        assert abs(c.data - 9.0) < 1e-6
        assert abs(a.grad - 6.0) < 1e-6  # d/da(a^2) = 2a = 6


class TestHelperFunctions:
    def test_softmax_sums_to_one(self):
        logits = [Value(1.0), Value(2.0), Value(3.0)]
        probs = softmax(logits)
        total = sum(p.data for p in probs)
        assert abs(total - 1.0) < 1e-6

    def test_softmax_argmax(self):
        logits = [Value(1.0), Value(5.0), Value(2.0)]
        probs = softmax(logits)
        assert probs[1].data > probs[0].data
        assert probs[1].data > probs[2].data

    def test_rmsnorm_scale(self):
        x = [Value(3.0), Value(4.0)]
        normed = rmsnorm(x)
        # RMS of [3, 4] = sqrt((9+16)/2) = sqrt(12.5) ~ 3.536
        # Each element should be scaled by 1/rms
        rms = math.sqrt(12.5)
        assert abs(normed[0].data - 3.0 / rms) < 1e-3
        assert abs(normed[1].data - 4.0 / rms) < 1e-3

    def test_linear(self):
        x = [Value(1.0), Value(2.0)]
        w = [[Value(1.0), Value(0.0)],
             [Value(0.0), Value(1.0)]]
        out = linear(x, w)
        assert abs(out[0].data - 1.0) < 1e-6  # 1*1 + 2*0
        assert abs(out[1].data - 2.0) < 1e-6  # 1*0 + 2*1

    def test_softmax_backward(self):
        logits = [Value(1.0), Value(2.0)]
        probs = softmax(logits)
        loss = -probs[1].log()
        loss.backward()
        # Gradients should be computed without error
        assert logits[0].grad != 0 or logits[1].grad != 0
