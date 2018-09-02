# -*- coding: utf-8 -*-
#
from krypy.linsys import LinearSystem
from krypy.deflation import DeflatedCg as KrypyCg

from .linear_operator import LinearOperator, wrap_linear_operator, wrap_inner_product


class Cg(object):
    def __init__(self, obj):
        self.MMlr0 = obj.MMlr0[:, 0]
        self.MMlr0_norm = obj.MMlr0_norm
        self.MlAMr = obj.MlAMr
        self.Mlr0 = obj.Mlr0[:, 0]
        self.flat_vecs = obj.flat_vecs
        self.store_arnoldi = obj.store_arnoldi
        self.maxiter = obj.maxiter
        self.iter = obj.iter
        self.explicit_residual = obj.explicit_residual
        self.resnorms = obj.resnorms
        self.tol = obj.tol
        self.x0 = obj.x0[:, 0]
        self.xk = obj.xk[:, 0]
        return

    def __repr__(self):
        string = "pykry CG object\n"
        string += "    MMlr0 = [{}, ..., {}]\n".format(self.MMlr0[0], self.MMlr0[-1])
        string += "    MMlr0_norm = {}\n".format(self.MMlr0_norm)
        string += "    MlAMr: {} x {} matrix\n".format(*self.MlAMr.shape)
        string += "    Mlr0: [{}, ..., {}]\n".format(self.Mlr0[0], self.Mlr0[-1])
        string += "    flat_vecs: {}\n".format(self.flat_vecs)
        string += "    store_arnoldi: {}\n".format(self.store_arnoldi)
        string += "    tol: {}\n".format(self.tol)
        string += "    maxiter: {}\n".format(self.maxiter)
        string += "    iter: {}\n".format(self.iter)
        string += "    explicit residual: {}\n".format(self.explicit_residual)
        string += "    resnorms: [{}, ..., {}]\n".format(
            self.resnorms[0], self.resnorms[-1]
        )
        string += "    x0: [{}, ..., {}]\n".format(self.x0[0], self.x0[-1])
        string += "    xk: [{}, ..., {}]".format(self.xk[0], self.xk[-1])
        return string


def cg(
    A,
    b,
    M=None,
    Minv=None,
    Ml=None,
    Mr=None,
    inner_product=None,
    exact_solution=None,
    x0=None,
    U=None,
    tol=1e-5,
    maxiter=None,
    use_explicit_residual=False,
    store_arnoldi=False,
):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    if isinstance(A, LinearOperator):
        A = wrap_linear_operator(A)

    if isinstance(M, LinearOperator):
        M = wrap_linear_operator(M)

    if isinstance(Ml, LinearOperator):
        Ml = wrap_linear_operator(Ml)

    if isinstance(Mr, LinearOperator):
        Mr = wrap_linear_operator(Mr)

    if inner_product:
        inner_product = wrap_inner_product(inner_product)

    # Make sure that the input vectors have two dimensions
    if U is not None:
        U = U.reshape(U.shape[0], -1)
    if x0 is not None:
        x0 = x0.reshape(U.shape[0], -1)

    linear_system = LinearSystem(
        A=A,
        b=b,
        M=M,
        Minv=Minv,
        Ml=Ml,
        ip_B=inner_product,
        # Setting those to `True` simply avoids a warning.
        self_adjoint=True,
        positive_definite=True,
        exact_solution=exact_solution,
    )
    out = KrypyCg(
        linear_system,
        x0=x0,
        U=U,
        tol=tol,
        maxiter=maxiter,
        explicit_residual=use_explicit_residual,
        store_arnoldi=store_arnoldi,
    )
    sol = Cg(out)
    return sol
