# AlgebraCore

AlgebraCore is a fast Python module for defining and computing with explicit
algebras.

It stays close to NumPy arrays while making the algebraic structure explicit:
named bases, coefficient elements, bilinear products via structure constants,
and linear transformations between those spaces.

The package is meant to be a small public kernel that is both useful on its own
and a fast foundation for richer layers.

## Core Features

- define any finite-dimensional algebra once you have chosen a basis and its
  structure constants
- treat many familiar algebras in one unified computational model
- keep the implementation dense and fast by storing coefficients and maps in
  NumPy arrays
- use algebraic syntax without hiding the product law behind a single overloaded
  element type
- allow multiple bilinear products on the same basis, instead of tying one
  basis to one privileged multiplication

The central point is that `AlgebraCore` is not limited to one algebra family.
Once a finite-dimensional algebra is expressed in a basis, it is represented in
the same way:

- a `Basis` for the labels
- an `Element` for coefficient vectors
- an `AlgebraProduct` for the bilinear multiplication law
- a `Transformation` for linear maps and basis changes

That is why complex numbers, matrix algebras, polynomial algebras,
quaternions, and user-defined algebras can all live in the same API.

The bundled `AlgebraCore.std` catalog is intentionally also a showcase. It is
there not only for convenience, but to make concrete that many things that look
very different on paper can still be treated as explicit algebras in the same
framework.

## Installation

Install the published package from PyPI:

```bash
pip install AlgebraCore
```

To use the latest repository state directly from GitHub:

```bash
pip install "git+https://github.com/henrik-schiller/AlgebraCore.git"
```

For local development:

```bash
pip install -e ".[test]"
```

Then run the test suite with:

```bash
pytest
```

The release procedure for PyPI is described in `RELEASING.md`.

## Many algebras, one module

The point of `AlgebraCore` is not only that you can define a single algebra
explicitly. It is that many different algebras can be handled in the same
computational form.

```python
from AlgebraCore.element import UnitElements
from AlgebraCore.std import (
    complex_basis,
    complex_product,
    so3_lie_bracket,
    so3_lie_basis,
    split_complex_basis,
    split_complex_product,
)

u_complex = UnitElements(complex_basis)
u_split = UnitElements(split_complex_basis)
u_so3 = UnitElements(so3_lie_basis)

z = 2 * u_complex.id + 3 * u_complex.i
w = -1 * u_complex.id + 4 * u_complex.i

a = 2 * u_split.id + 1 * u_split.j
b = -1 * u_split.id + 3 * u_split.j

x = 2 * u_so3.e1 + 1 * u_so3.e2
y = -1 * u_so3.e2 + 4 * u_so3.e3

print(z @ complex_product @ w)              # -14*id + 5*i
print(a @ split_complex_product @ b)        # 1*id + 5*j
print(x @ so3_lie_bracket @ y)              # 4*e1 + -8*e2 + -2*e3
```

This is the main idea behind the module: complex numbers, split-complex
numbers, matrix algebras, polynomial algebras, quaternions, octonions, and
simple Lie-algebra examples all fit into the same explicit basis-plus-product
framework.

## Quick example

```python
from AlgebraCore.element import UnitElements
from AlgebraCore.std import complex_basis, complex_product

basis = complex_basis
product = complex_product
u = UnitElements(basis)

z = 2 * u.id + 3 * u.i
w = -1 * u.id + 4 * u.i

print(z @ product @ w)  # -14*id + 5*i
```

## Explicit products

Mathematics often uses different symbols for different bilinear laws. In code,
`AlgebraCore` makes the product itself explicit and uses one consistent form:

```python
a @ product @ b
```

This does not erase the mathematical differences between those products. It
makes the choice of product programmable, inspectable, and swappable while the
underlying data stays fast and dense in NumPy arrays.

This is important because a basis and a product are not the same thing. The
same basis may support several different bilinear laws. For example, one and
the same coordinate basis can carry Clifford, exterior, and interior products.
`AlgebraCore` keeps that choice explicit instead of baking one multiplication
directly into the element type.

| Math idea | Typical notation | AlgebraCore |
| --- | --- | --- |
| matrix product | `AB` | `A @ matrix_product @ B` |
| polynomial product | `p(x)q(x)` | `p @ polynomial_product @ q` |
| Clifford product | `ab` | `a @ clifford_product @ b` |
| exterior product | `a ∧ b` | `a @ wedge_product @ b` |
| Lie bracket | `[a, b]` | `a @ lie_bracket @ b` |

Instead of hiding multiplication inside `Element`, `AlgebraCore` treats the
product itself as a first-class object. That keeps scalar scaling and algebra
multiplication separate:

- `scalar * element` means scalar scaling
- `a @ product @ b` means multiplication with an explicit bilinear law
- `transformation @ element` means linear application

The point is deliberate: many algebras look different on paper, but once a
basis and a bilinear law are fixed, they can be handled in one computational
form.

## Define Your Own Algebra

Defining your own algebra is a core feature of `AlgebraCore`.

The intended input format is plain Python plus NumPy, not a custom DSL and not
YAML. In practice, you provide:

- a basis
- a dense tensor `C[i, j, k]` of structure constants

and construct an `AlgebraProduct`.

```python
import numpy as np

from AlgebraCore.basis import Basis
from AlgebraCore.element import UnitElements
from AlgebraCore.product import AlgebraProduct

basis = Basis(["id", "s"])
C = np.zeros((2, 2, 2), dtype=float)

# id is the multiplicative identity
C[0, :, :] = np.eye(2)
C[:, 0, :] = np.eye(2)

# custom rule: s * s = id + s
C[1, 1, 0] = 1.0
C[1, 1, 1] = 1.0

product = AlgebraProduct(basis, C)
u = UnitElements(basis)

result = (2 * u.id + 3 * u.s) @ product @ (-1 * u.id + 4 * u.s)
print(result)
```

So yes: custom algebras given by structure constants are directly supported.

## What is inside

- `Basis` and `TensorBasis`
- `Element` and `UnitElements`
- `AlgebraProduct` and tensor products of products
- `Transformation`
- a curated standard catalog under `AlgebraCore.std`

The standard catalog currently includes familiar dense examples such as
complex, dual, split-complex, matrix, polynomial, quaternion, octonion, and
simple Lie-algebra examples.

For the fixed canonical examples in `AlgebraCore.std`, the preferred style is
to use the exported basis and product objects directly, for example
`complex_basis`, `complex_product`, or `so3_lie_bracket`. Those are convenient
canonical representatives, not a claim that a basis admits only one product.

## About TensorBasis

`TensorBasis` is a small computational helper name, not a standard mathematical
term that readers are expected to already know.

What it does is simple: it takes several factor bases and builds the basis for
their tensor product by concatenating their axes. The result is still just a
regular `Basis`.

```python
TensorBasis([Basis(["a", "b"]), Basis(["x", "y"])])
```

produces the same kind of object as `Basis([["a", "b"], ["x", "y"]])`.

The helper exists because this is exactly what is needed computationally for
outer products of elements, Kronecker-style tensor products of transformations,
and tensor products of algebra products. In other words, the name is chosen for
programming clarity, not because `TensorBasis` is meant to introduce a new
mathematical object beyond `Basis`.

## Design Notes

For a slightly more explicit discussion of the main API choices, see
`DESIGN.md`.

## Status

`AlgebraCore` is intended to be the public, maintainable kernel of the project.
APIs may still sharpen, but the scope is deliberately narrow.
