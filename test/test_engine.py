import torch, random
from autograd import nn
from autograd.engine import Value
from utils import draw_dot

def test_check_one():
    # scalarflow implementation
    x1 = Value(2.0, label='x1')
    w1 = Value(-3.0, label='w1')
    x2 = Value(0.0, label='x2')
    w2 = Value(1.0, label='w2')
    b = Value(6.88137, label='b')
    n = x1*w1 + x2*w2 + b

    e = (2*n).exp()
    o = (e-1) / (e+1); o.label = 'o'
    o.backward()
    x1sf, x2sf, ysf = x1, x2, o

    # pytorch implementation
    x1 = torch.Tensor([2.0]).double(); x1.requires_grad=True
    w1 = torch.Tensor([-3.0]).double(); w1.requires_grad=True
    x2 = torch.Tensor([0.0]).double(); x2.requires_grad=True
    w2 = torch.Tensor([1.0]).double(); w2.requires_grad=True
    b = torch.Tensor([6.88137]).double(); b.requires_grad=True
    n = x1*w1 + x2*w2 + b
    o = torch.tanh(n)
    o.backward()
    x1pt, x2pt, ypt = x1, x2, o

    tol = 1e-6
    # forward pass matches
    assert abs(ysf.data - ypt.data.item()) < tol
    # backward pass matches
    assert abs(x1sf.grad - x1pt.grad.item()) < tol
    assert abs(x2sf.grad - x2pt.grad.item()) < tol
    print(f'test one successful!')

def test_check_two():
    # scalarflow implementation
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    asf, bsf, gsf = a, b, g

    # pytorch implementation
    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gsf.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(asf.grad - apt.grad.item()) < tol
    assert abs(bsf.grad - bpt.grad.item()) < tol
    print(f'test two successful!')

def test_check_utils():
    # # a very simple example
    # x = Value(1.0)
    # y = (x * 2 + 1).relu()
    # y.backward()
    # dot = draw_dot(y)
    # # Save the graph to a file
    # dot.render(filename='sf_computation_graph', format='png', view=True) 

    # a simple 2D neuron
    random.seed(1337)
    n = nn.Neuron(2)
    x = [Value(1.0), Value(-2.0)]
    y = n(x)
    y.backward()
    dot = draw_dot(y)
    # Save the graph to a file
    dot.render(filename='sf_computation_graph', format='png', view=True) 

if __name__ == "__main__":
    test_check_one()
    test_check_two()
    test_check_utils()
