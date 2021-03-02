from algebra import *
from numpy import pi


def identity(n_bits):
    """identity given number of q-bits"""
    return Operator(np.identity(2**n_bits))


I = identity(1)
H = Operator([[1, 1], [1, -1]]).n()

X = Operator([[0, 1], [1, 0]])
Y = Operator([[0, -1j], [1j, 0]])
Z = Operator([[1, 0], [0, -1]])

NOT = X
CX = CNOT = C_(NOT)

R = phase_shift
S = phase_shift(pi / 2)
T = phase_shift(pi / 8)

Toffoli = CCNOT = C_(C_(NOT))

SWAP = Operator([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])

Fredkin = CSWAP = C_(SWAP)

# complex number: (1+i)/2
q_ = complex(1, 1)/2


NOT_sqrt = I * q_ + X * q_.conjugate()

SWAP_sqrt = stack(Id(1), NOT_sqrt, Id(1))


def ising(mat):
    """unofficial Ising matrices"""
    def ising_(angle):
        a = np.cos(angle) * tensor_prod(I, I)
        b = -1j * np.sin(angle) * tensor_prod(mat, mat)
        return Operator(a + b)
    return ising_


Ising_XX = ising(X)
Ising_YY = ising(Y)


def Ising_ZZ(angle):
    a = np.cos(angle/2) * tensor_prod(I, I)
    b = 1j * np.sin(angle/2) * tensor_prod(Z, Z)
    return Operator(a + b)


del q_
