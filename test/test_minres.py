# -*- coding: utf-8 -*-
#
import numpy

import pykry


def test_minres():
    A = numpy.diag([1.0e-3] + list(range(2, 101)))
    b = numpy.ones(100)

    out = pykry.minres(A, b)
    print(out)

    ref = 1004.187372488912
    assert abs(numpy.sum(numpy.abs(out.xk)) - ref) < 1.0e-12 * ref
    ref = 1000.0003124632159
    assert abs(numpy.sqrt(numpy.dot(out.xk, out.xk)) - ref) < 1.0e-12 * ref
    ref = 999.9999949713145
    assert abs(numpy.max(numpy.abs(out.xk)) - ref) < 1.0e-12 * ref
    return


def test_minres_linear_operator():
    n = 100
    A = numpy.diag([1.0e-3] + list(range(2, n + 1)))
    b = numpy.ones(n)

    linear_operator = pykry.LinearOperator(
        (n, n), float, dot=lambda x: A.dot(x), dot_adj=lambda x: A.dot(x)
    )

    out = pykry.minres(linear_operator, b)

    ref = 1004.187372488912
    assert abs(numpy.sum(numpy.abs(out.xk)) - ref) < 1.0e-12 * ref
    ref = 1000.0003124632159
    assert abs(numpy.sqrt(numpy.dot(out.xk, out.xk)) - ref) < 1.0e-12 * ref
    ref = 999.9999949713145
    assert abs(numpy.max(numpy.abs(out.xk)) - ref) < 1.0e-12 * ref
    return
