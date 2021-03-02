from utils import *
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from math import sin, cos, pi, tanh
import numpy as np


def norm(v):
    return sum(np.linalg.norm(i) ** 2 for i in v)


class Vector:
    def __init__(self, v):
        if norm(v) == 0:
            raise ValueError('Vectors must be normalizable!')
        self.v = np.array(v) / norm(v) ** .5

    def __mul__(self, other):
        if type(other) == AdjointVector:
            return Operator(np.outer(self.v, other.v))
        elif type(other) == Operator:
            raise TypeError
        elif type(other) == Vector:
            raise TypeError
        else:
            return Vector(self.v * other)

    def __rmul__(self, other):
        return Vector(self.v * other)

    def __add__(self, other):
        if type(other) == Vector:
            return Vector(self.v + other.v)
        else:
            raise ValueError

    def __sub__(self, other):
        if type(other) == Vector:
            return Vector(self.v - other.v)
        else:
            raise ValueError

    def __neg__(self):
        return Vector(-self.v)

    def __radd__(self, other):
        return self

    def __repr__(self):
        return f'Vector({self.v})'

    def __iter__(self):
        return iter(self.v)

    def __len__(self):
        return len(self.v)

    def __getitem__(self, item):
        return self.v[item]

    def __eq__(self, other, tolerance=10**-14):
        if type(other) == Vector:
            return (self - other).norm() <= tolerance
        else:
            return False

    def plot(self, title=''):
        plt.title(title)
        plt.ylim(0, 1)
        v = [np.linalg.norm(i)**2 for i in self.v]
        x = list(range(len(v)))
        plt.bar(x, v)
        plt.show()

    def choose(self):
        return choose([np.linalg.norm(i)**2 for i in self.v])


class AdjointVector:
    def __init__(self, v):
        self.v = np.array(v).conjugate()

    def __mul__(self, other):
        if type(other) == AdjointVector:
            raise TypeError
        elif type(other) == Operator:
            return AdjointVector(self.v.dot(other.mat))
        elif type(other) == Vector:
            return self.v.dot(other.v)
        else:
            return Vector(self.v * other)

    def __rmul__(self, other):
        return Vector(self.v * other)

    def __radd__(self, other):
        return self

    def __repr__(self):
        return f'Adjoint({self.v})'


def plot_complex(axes, length, angle, x0, y0, scale=3, pt_size=.1, h=.4, v=.85):

    color = hsv_to_rgb(((-angle/(2*pi)-h) % 1, tanh(length), v))

    circle = plt.Circle((x0 * scale, y0 * scale), length, color=color, fill=False)
    point = (x0 * scale + length * cos(angle), y0 * scale + length * sin(angle))
    circle2 = plt.Circle(point, pt_size, color=color)

    axes.add_artist(circle)
    axes.add_artist(circle2)


class Operator:

    def __init__(self, mat):
        if type(mat) == Operator:
            self.mat = mat.mat
        else:
            self.mat = np.array(mat)
        self.rows = len(self.mat)
        self.col = len(self.mat[0])

    def __mul__(self, other):
        if type(other) == AdjointVector:
            raise TypeError
        elif type(other) == Operator:
            return Operator(self.mat.dot(other.mat))
        elif type(other) == Vector:
            return Vector(self.mat.dot(other.v))
        else:  # scalar
            return Operator(self.mat * other)

    def __add__(self, other):
        return Operator(self.mat + other.mat)

    def __sub__(self, other):
        return Operator(self.mat - other.mat)

    def __truediv__(self, other):
        return Operator(self.mat / other)

    def __rmul__(self, other):  # careful with vectors
        return self * other

    def __radd__(self, other):
        return self

    def __repr__(self):
        m = [[str_complex(j) for j in i] for i in self.mat]
        length = max([max(len(j) for j in i) for i in m]) + 2

        s = ''
        for i in m:
            for j in i:
                s += f'{j:{length}}'
            s += '\n'
        return s

    def __len__(self):
        return len(self.mat)

    def __pow__(self, power: int, modulo=None):
        if power <= 0:
            return Id(len(self.mat))
        elif power == 1:
            return self
        else:
            return self * self**(power-1)

    def __getitem__(self, item):
        return self.mat[item]

    def __neg__(self):
        return Operator(-self.mat)

    def norm(self):
        return np.linalg.norm(self.mat) / len(self) ** .5

    def n(self):
        """normalized operator"""
        return Operator(self.mat / self.norm())

    def t(self):
        """transpose"""
        return Operator(self.mat.transpose())

    def conj(self):
        """complex conjugate"""
        return Operator(self.mat.conj())

    def adj(self):
        """adjoint"""
        return self.t().conj()

    def tensor(self, other):
        """tensor product"""
        return Operator(np.kron(self.mat, other.mat))

    def __xor__(self, other):
        return self.tensor(other)

    def stack(self, other):
        m = np.block([[self.mat, np.zeros((self.rows, other.col))],
                      [np.zeros((other.rows, self.col)), other.mat]])
        return Operator(m)

    def plot(self, scale=3, pt_size=.2, h=.4, v=.93, title='', skip_zeros=False):
        """plot complex matrix"""
        y_len = len(self.mat)
        x_len = len(self.mat[0])

        figure, axes = plt.subplots()

        axes.axis('off')

        plt.title(title)
        plt.xlim(-scale / 2, scale * (x_len - 1) + scale / 2)
        plt.ylim(-scale / 2, scale * (y_len - 1) + scale / 2)
        plt.xticks([])
        plt.yticks([])

        axes.set_aspect(1)
        for j in range(y_len):
            for i in range(x_len):
                z = self.mat[j][i]
                m = np.linalg.norm(z)
                if not (skip_zeros and not m):
                    a = np.angle(z)
                    plot_complex(axes, m, a, i, y_len - 1 - j, scale=scale, pt_size=pt_size, h=h, v=v)

        plt.show()

    def __eq__(self, other, tolerance=10**-14):
        if type(other) == Operator:
            return (self - other).norm() <= tolerance
        else:
            return False

    def n_qbits(self):
        n = len(self.mat)
        out = 0
        while True:
            n, r = n // 2, n % 2
            if n == 0:
                return out
            out += 1
            if r > 0:
                return None


def tensor_prod(*args: Operator) -> Operator:
    """tensor product"""
    out = args[0]
    for m in args[1:]:
        out = out.tensor(m)
    return out


def stack(*args: Operator) -> Operator:
    out = args[0]
    for m in args[1:]:
        out = out.stack(m)
    return out


def blocks(mat) -> Operator:
    m = [[j.mat for j in i] for i in mat]
    return Operator(np.block(m))


def phase_shift(phase):
    return Operator([[1, 0], [0, np.exp(np.complex(0, phase))]])


def commutator(a, b):
    return a * b - b * a


def Id(size):
    """identity given matrix size"""
    if size is not None:
        return Operator(np.identity(size))


def C_(mat):
    i = Id(len(mat))
    return stack(i, mat)


def unitary_vector(n, size):
    v = np.zeros(size)
    v[n] = 1
    return Vector(v)


def bits_to_vector_space(*bits):
    """transform bits into vector
    00 -> 1000
    01 -> 0100
    10 -> 0010
    11 -> 0001
    """
    n = sum(2 ** i * bits[len(bits) - 1 - i] for i in range(len(bits)))
    v = unitary_vector(n, size=2**len(bits))
    return v


def vector_space_to_bits(v):
    """transform vector into bits
    1000 -> 00
    0100 -> 01
    0010 -> 10
    0001 -> 11
    """
    def gen(i, m):
        if m > 1:
            if i < m//2:
                return [0] + gen(i, m//2)
            else:
                return [1] + gen(i-m//2, m//2)
        else:
            return []

    return gen(np.argmax(v), len(v))


def gen_all_n_bit_combo(n):
    """"""
    if n > 0:
        for i in range(2):
            for j in gen_all_n_bit_combo(n-1):
                yield (i,) + j
    else:
        yield tuple()


def custom_operator(fun, n_bits):
    mat = [[fun(i, j) for j in gen_all_n_bit_combo(n_bits)]
           for i in gen_all_n_bit_combo(n_bits)]
    return Operator(mat)


def sandwich(*operators) -> Operator:
    out = operators[0]
    for o in operators[1:]:
        out *= o
    for o in operators[-2::-1]:
        out *= o.adj()
    return out
