# python-project-scaffold

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/pykry/master.svg)](https://circleci.com/gh/nschloe/pykry/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/pykry.svg)](https://codecov.io/gh/nschloe/pykry)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPi Version](https://img.shields.io/pypi/v/pykry.svg)](https://pypi.org/project/pykry)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/pykry.svg?logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/pykry)

Some description.

Run
```
find . -type f -print0 | xargs -0 sed -i 's/pykry/your-project-name/g'
```
to customize.

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
