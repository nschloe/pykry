# -*- coding: utf-8 -*-
#
import numpy

import pykry


def test_cg():
    A = numpy.diag([1.0e-3] + list(range(2, 101)))
    b = numpy.ones(100)

    out = pykry.cg(A, b)
    print(out)

    ref = 1004.1873775173957
    assert abs(numpy.sum(numpy.abs(out.xk)) - ref) < 1.0e-12 * ref
    ref = 1000.0003174916551
    assert abs(numpy.sqrt(numpy.dot(out.xk, out.xk)) - ref) < 1.0e-12 * ref
    ref = 999.9999999997555
    assert abs(numpy.max(numpy.abs(out.xk)) - ref) < 1.0e-12 * ref
    return


def test_cg_linear_operator():
    n = 100
    A = numpy.diag([1.0e-3] + list(range(2, n + 1)))
    b = numpy.ones(n)

    linear_operator = pykry.LinearOperator(
        (n, n), float, dot=lambda x: A.dot(x), dot_adj=lambda x: A.dot(x)
    )

    out = pykry.cg(linear_operator, b)

    ref = 1004.1873775173957
    assert abs(numpy.sum(numpy.abs(out.xk)) - ref) < 1.0e-12 * ref
    ref = 1000.0003174916551
    assert abs(numpy.sqrt(numpy.dot(out.xk, out.xk)) - ref) < 1.0e-12 * ref
    ref = 999.9999999997555
    assert abs(numpy.max(numpy.abs(out.xk)) - ref) < 1.0e-12 * ref
    return
