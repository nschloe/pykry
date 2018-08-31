# -*- coding: utf-8 -*-
#
from krypy.utils import LinearOperator as KrypyLinearOperator


class LinearOperator(object):
    def __init__(self, shape, dtype, dot=None, dot_adj=None):
        self.shape = shape
        self.dtype = dtype
        self.dot = dot
        self.dot_adj = dot_adj
        return

    def __mul__(self, X):
        return self.dot(X)


def wrap(linear_operator):
    """Wrap a pykry LinearOperator in a KryPy LinearOperator. This is essentially just
    reshaping.
    """

    def dot(X):
        assert X.shape[1] == 1
        out = linear_operator.dot(X[:, 0])
        return out.reshape(-1, 1)

    def dot_adj(X):
        assert X.shape[1] == 1
        out = linear_operator.dot_adj(X[:, 0])
        return out.reshape(-1, 1)

    return KrypyLinearOperator(
        linear_operator.shape, linear_operator.dtype, dot, dot_adj
    )
