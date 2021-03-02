import numpy as np
from numpy import pi
from random import random


def np_equal(a, b, precision=10**-15):
    return abs(a-b) <= precision


def str_simplify(s):
    while s[-1] == '0':
        s = s[:-1]
    if s[-1] == '.':
        s = s[:-1]
    return s


def str_complex(z, dec=3):
    if np_equal(z, 0):
        return ' .'
    elif np_equal(np.imag(z), 0):
        s = str_simplify(f'{np.real(z):+.{dec}f}')
        s = s.replace('+', ' ')
        return s
    elif np_equal(np.real(z), 0):
        s = str_simplify(f'{np.imag(z):+.{dec}f}')
        s = s.replace('+', ' ') + ' i'
        s = s.replace('1 ', '')
        return s
    else:
        m = np.abs(z)
        a = np.angle(z) / pi
        m = f'{m:.{dec}f}'
        a = f'{a:.{dec}f}'
        return f'{str_simplify(m)} ∠ {str_simplify(a)} π'


def choose(v):
    x = random()
    for i in range(len(v)):
        x -= v[i]
        if x < 0:
            return i
