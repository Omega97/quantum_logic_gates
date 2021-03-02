from quantum_gates import *


def test_1():

    A = X ^ Y ^ X
    ht = H ^ H ^ H

    O = sandwich(ht, A)

    # o.plot(skip_zeros=True)
    B = (Z ^ Y ^ Z).conj()
    # B.plot(skip_zeros=True)

    print((B - O).norm())
    print(B == O)

    # v = bits_to_vector_space(0, 0, 0)
    # (o * v).plot()


def test_2():
    A = H ^ H
    B = Y ^ Z

    O = sandwich(A, B)
    O.plot()


def test_3():
    i = complex(0, 1)
    h = (i / 2**(1/2))**(1/2)

    # A = Operator([[1, h], [h, -i]])
    A = h.conjugate() * I + h * H

    # (A**2).plot()
    print(A**2)



if __name__ == "__main__":
    # test_1()
    # test_2()
    test_3()
