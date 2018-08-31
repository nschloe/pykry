# -*- coding: utf-8 -*-
#
import numpy

import pykry


def test_pykry():
    A = numpy.diag([1.0e-3] + list(range(2, 101)))
    b = numpy.ones(100)

    out = pykry.gmres(A, b)

    ref = 1004.1873724888546
    assert abs(numpy.sum(numpy.abs(out.xk)) - ref) < 1.0e-14 * ref
    ref = 1000000.6249262823
    assert abs(numpy.dot(out.xk, out.xk) - ref) < 1.0e-14 * ref
    ref = 999.999994971191
    assert abs(numpy.max(numpy.abs(out.xk)) - ref) < 1.0e-14 * ref
    return
