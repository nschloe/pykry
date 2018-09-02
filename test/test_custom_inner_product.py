# -*- coding: utf-8 -*-
#
import numpy

import pykry


def test_custom_inner_product():
    n = 100
    A = numpy.diag([1.0e-3] + list(range(2, n + 1)))
    b = numpy.ones(n)

    def inner(a, b):
        return numpy.dot(a, b)

    out = pykry.cg(A, b, inner_product=inner)

    ref = 1004.1873775173957
    assert abs(numpy.sum(numpy.abs(out.xk)) - ref) < 1.0e-12 * ref
    ref = 1000.0003174916551
    assert abs(numpy.sqrt(numpy.dot(out.xk, out.xk)) - ref) < 1.0e-12 * ref
    ref = 999.9999999997555
    assert abs(numpy.max(numpy.abs(out.xk)) - ref) < 1.0e-12 * ref
    return
