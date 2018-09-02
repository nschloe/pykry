# -*- coding: utf-8 -*-
#
import numpy
import pytest

import pykry


@pytest.mark.parametrize("method, ref", [
    (pykry.cg, [1004.1873775173957, 1000.0003174916551, 999.9999999997555]),
    (pykry.minres, [1004.187372488912, 1000.0003124632159, 999.9999949713145]),
    (pykry.gmres, [1004.1873724888546, 1000.0003124630923, 999.999994971191])
])
def test_matrix(method, ref):
    A = numpy.diag([1.0e-3] + list(range(2, 101)))
    b = numpy.ones(100)

    out = method(A, b)
    print(out)

    assert abs(numpy.sum(numpy.abs(out.xk)) - ref[0]) < 1.0e-12 * ref[0]
    assert abs(numpy.sqrt(numpy.dot(out.xk, out.xk)) - ref[1]) < 1.0e-12 * ref[1]
    assert abs(numpy.max(numpy.abs(out.xk)) - ref[2]) < 1.0e-12 * ref[2]
    return


@pytest.mark.parametrize("method, ref", [
    (pykry.cg, [1004.1873775173957, 1000.0003174916551, 999.9999999997555]),
    (pykry.minres, [1004.187372488912, 1000.0003124632159, 999.9999949713145]),
    (pykry.gmres, [1004.1873724888546, 1000.0003124630923, 999.999994971191])
])
def test_linear_operator(method, ref):
    n = 100
    A = numpy.diag([1.0e-3] + list(range(2, n + 1)))
    b = numpy.ones(n)

    linear_operator = pykry.LinearOperator(
        (n, n), float, dot=lambda x: A.dot(x), dot_adj=lambda x: A.dot(x)
    )

    out = method(linear_operator, b)
    print(out)

    assert abs(numpy.sum(numpy.abs(out.xk)) - ref[0]) < 1.0e-12 * ref[0]
    assert abs(numpy.sqrt(numpy.dot(out.xk, out.xk)) - ref[1]) < 1.0e-12 * ref[1]
    assert abs(numpy.max(numpy.abs(out.xk)) - ref[2]) < 1.0e-12 * ref[2]
    return
