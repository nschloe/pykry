# -*- coding: utf-8 -*-
#
import numpy

import pykry


def test_pykry():
    A = numpy.diag([1.0e-3] + list(range(2, 101)))
    b = numpy.ones(100)

    sol = pykry.gmres(A, b)
    print(sol)
    exit(1)

    # # plot residuals
    # from matplotlib import pyplot
    # pyplot.semilogy(sol.resnorms)
    # pyplot.show()
    return
