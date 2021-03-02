from algebra import unitary_vector


def measure_once_for_each_input(op):
    out = []
    for i in range(len(op)):
        v = op * unitary_vector(i, len(op))
        out += [v.choose()]
    return out
