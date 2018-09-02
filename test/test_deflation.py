# -*- coding: utf-8 -*-
#
import numpy
import pytest

import pykry


@pytest.mark.parametrize(
    "method, ref",
    [
        (pykry.cg, [1004.1873775173271, 1000.0003174918709, 1000.0]),
        (pykry.minres, [1004.1873774950692, 1000.0003174918709, 1000.0]),
        (pykry.gmres, [1004.1873774950692, 1000.0003174918709, 1000.0]),
    ],
)
def test_gmres(method, ref):
    n = 100
    A = numpy.diag([1.0e-3] + list(range(2, n + 1)))
    b = numpy.ones(n)

    # deflate out the vector that belongs to the small eigenvalue
    U = numpy.zeros(n)
    U[0] = 1.0
    out = method(A, b, U=U)

    assert abs(numpy.sum(numpy.abs(out.xk)) - ref[0]) < 1.0e-12 * ref[0]
    assert abs(numpy.sqrt(numpy.dot(out.xk, out.xk)) - ref[1]) < 1.0e-12 * ref[1]
    assert abs(numpy.max(numpy.abs(out.xk)) - ref[2]) < 1.0e-12 * ref[2]
    return
