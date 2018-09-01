# -*- coding: utf-8 -*-
#
from krypy.linsys import LinearSystem, Gmres as KrypyGmres

from .linear_operator import wrap, LinearOperator


class Gmres(object):
    def __init__(self, obj):
        self.MMlr0 = obj.MMlr0[:, 0]
        self.MMlr0_norm = obj.MMlr0_norm
        self.MlAMr = obj.MlAMr
        self.Mlr0 = obj.Mlr0[:, 0]
        self.R = obj.R
        self.V = obj.V
        self.flat_vecs = obj.flat_vecs
        self.store_arnoldi = obj.store_arnoldi
        self.ortho = obj.ortho
        self.maxiter = obj.maxiter
        self.iter = obj.iter
        self.explicit_residual = obj.explicit_residual
        self.resnorms = obj.resnorms
        self.tol = obj.tol
        self.x0 = obj.x0[:, 0]
        self.xk = obj.xk[:, 0]
        return

    def __repr__(self):
        string = "pykry GMRES object\n"
        string += "    MMlr0 = [{}, ..., {}]\n".format(self.MMlr0[0], self.MMlr0[-1])
        string += "    MMlr0_norm = {}\n".format(self.MMlr0_norm)
        string += "    MlAMr: {} x {} matrix\n".format(*self.MlAMr.shape)
        string += "    Mlr0: [{}, ..., {}]\n".format(self.Mlr0[0], self.Mlr0[-1])
        string += "    R: {} x {} matrix\n".format(*self.R.shape)
        string += "    V: {} x {} matrix\n".format(*self.V.shape)
        string += "    flat_vecs: {}\n".format(self.flat_vecs)
        string += "    store_arnoldi: {}\n".format(self.store_arnoldi)
        string += "    ortho: {}\n".format(self.ortho)
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


def gmres(
    A,
    b,
    M=None,
    Minv=None,
    Ml=None,
    Mr=None,
    inner_product=None,
    exact_solution=None,
    ortho="mgs",
    x0=None,
    tol=1e-5,
    maxiter=None,
    use_explicit_residual=False,
    store_arnoldi=False,
    dtype=None,
):
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == b.shape[0]

    if isinstance(A, LinearOperator):
        A = wrap(A)

    linear_system = LinearSystem(
        A=A,
        b=b,
        M=M,
        Minv=Minv,
        Ml=Ml,
        ip_B=inner_product,
        exact_solution=exact_solution,
    )
    out = KrypyGmres(
        linear_system,
        ortho=ortho,
        x0=x0,
        tol=tol,
        maxiter=maxiter,
        explicit_residual=use_explicit_residual,
        store_arnoldi=store_arnoldi,
        dtype=dtype,
    )
    sol = Gmres(out)
    return sol
