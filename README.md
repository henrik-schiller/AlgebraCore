# AlgebraCore

AlgebraCore is the small, dense, NumPy-based core behind the broader Algebra
stack.

It provides explicit algebra objects that stay close to arrays while adding the
structure NumPy does not model directly: named bases, coefficient elements,
bilinear products via structure constants, and linear transformations between
those spaces.

This package is meant to stay small and comparatively stable. It is the part
that can be published independently and used as a fast foundation for more
specialized layers.

## What is inside

- `Basis` and `TensorBasis`
- `Element` and `UnitElements`
- `AlgebraProduct` and tensor products of products
- `Transformation`
- a small stable standard catalog under `AlgebraCore.std`

The standard catalog currently includes familiar dense examples such as complex,
dual, matrix, polynomial, and quaternion algebras.

## Installation

```bash
pip install -e .
```

For tests:

```bash
pip install -e ".[test]"
pytest
```

## Quick example

```python
from AlgebraCore.element import UnitElements
from AlgebraCore.std import complex_basis, complex_product

basis = complex_basis()
product = complex_product(basis)
u = UnitElements(basis)

z = 2 * u.id + 3 * u.i
w = -1 * u.id + 4 * u.i

print(z @ product @ w)  # -14*id + 5*i
```

## Status

`AlgebraCore` is intended to be the public, maintainable kernel of the project.
APIs may still sharpen, but the scope is deliberately narrow.
