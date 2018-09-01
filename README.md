# python-project-scaffold

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/pykry/master.svg)](https://circleci.com/gh/nschloe/pykry/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/pykry.svg)](https://codecov.io/gh/nschloe/pykry)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPi Version](https://img.shields.io/pypi/v/pykry.svg)](https://pypi.org/project/pykry)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/pykry.svg?logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/pykry)

pykry is a thin wrapper around [KryPy](https://github.com/andrenarchy/krypy) that makes
using Krylov subspace methods in Python a little more convenient. Simply create the
matrix and the right-hand side, then fire it up:
```python
import numpy
import pykry

A = numpy.diag([1.0e-3] + list(range(2, 101)))
b = numpy.ones(100)

# out = pykry.cg(A, b)
# out = pykry.minres(A, b)
out = pykry.gmres(A, b)

# out.xk contains the last iterate (ideally the solution),
# out.resnorms the relative residual norms;
# there's plenty more
```
![convergence](https://nschloe.github.io/pykry/conv.png)

Owing to KryPy, pykry has a plethora of extra parameters to hand to either one of the
methods.

Getting more fancy with linear operators is as easy as defining a matrix-vector
multiplication:
```python
import numpy
import pykry

n = 100
A = numpy.diag([1.0e-3] + list(range(2, n + 1)))

def dot(x):
  return A.dot(x)

linear_operator = pykry.LinearOperator((n, n), float, dot=lambda x: dot, dot_adj=dot)
b = numpy.ones(n)
out = pykry.cg(linear_operator, b)
```



### Installation

pykry is [available from the Python Package
Index](https://pypi.org/project/pykry/), so simply type
```
pip install -U pykry
```
to install or upgrade.

### Testing

To run the pykry unit tests, check out this repository and type
```
pytest
```

### Distribution

To create a new release

1. bump the `__version__` number,

2. publish to PyPi and GitHub:
    ```
    make publish
    ```

### License

pykry is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
