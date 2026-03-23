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

## Core Features

- define any finite-dimensional algebra once you have chosen a basis and its
  structure constants
- treat many familiar algebras in one unified computational model
- keep the implementation dense and fast by storing coefficients and maps in
  NumPy arrays
- use algebraic syntax without hiding the product law behind a single overloaded
  element type

The central point is that `AlgebraCore` is not limited to one algebra family.
Once a finite-dimensional algebra is expressed in a basis, it is represented in
the same way:

- a `Basis` for the labels
- an `Element` for coefficient vectors
- an `AlgebraProduct` for the bilinear multiplication law
- a `Transformation` for linear maps and basis changes

That is why complex numbers, matrix algebras, polynomial algebras,
quaternions, and user-defined algebras can all live in the same API.

## One notation for many products

Mathematics often uses different symbols for different bilinear laws. In code,
`AlgebraCore` makes the product itself explicit and uses one consistent form:

```python
a @ product @ b
```

This does not erase the mathematical differences between those products. It
makes the choice of product programmable, inspectable, and swappable while the
underlying data stays fast and dense in NumPy arrays.

| Math idea | Typical notation | AlgebraCore |
| --- | --- | --- |
| matrix product | `AB` | `A @ matrix_product @ B` |
| polynomial product | `p(x)q(x)` | `p @ polynomial_product @ q` |
| Clifford product | `ab` | `a @ clifford_product @ b` |
| exterior product | `a ∧ b` | `a @ wedge_product @ b` |
| Lie bracket | `[a, b]` | `a @ lie_product @ b` |

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
- a small stable standard catalog under `AlgebraCore.std`

The standard catalog currently includes familiar dense examples such as complex,
dual, matrix, polynomial, and quaternion algebras.

## Why This Is A Useful Niche

`AlgebraCore` sits in a specific place between low-level numerical libraries and
specialized symbolic or geometric-algebra packages.

- compared with plain NumPy, it keeps the algebraic structure explicit instead
  of leaving you with unnamed arrays and hand-written multiplication logic
- compared with more specialized algebra packages, it is not restricted to one
  family such as geometric algebra
- compared with more symbolic systems, it stays fast by working directly with
  dense NumPy arrays

So the niche is: explicit, programmable, finite-dimensional algebras with a
clean algebraic API and a dense numerical backend.

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

## Transformation Conventions

`Transformation` is used in two closely related but not identical ways, and it
is worth stating that explicitly.

- `tf @ elem` is the primary computational application operator. It contracts
  the stored transformation tensor with the element coefficients.
- `elem.transform(tf)` is the basis-change helper. It applies the inverse of the
  flattened transformation matrix.

This distinction exists because the library needs both viewpoints:

- a transformation as an explicit multilinear array that can be contracted,
  tensored, and composed
- a transformation as a change-of-basis object between coordinate systems

If you are using `Transformation` as a linear operator in computations, prefer
`tf @ elem`. If you are explicitly changing coordinates from one basis
description to another, prefer `elem.transform(tf)`.

## Why `a @ product @ b`?

In AlgebraCore, the multiplication law is explicit.

Instead of baking a single product into the `Element` type, the product itself
is treated as a first-class object. This matters because the same basis can
support different bilinear laws: for example matrix multiplication, polynomial
multiplication, Clifford products, exterior products, interior products, or
user-defined products.

For that reason, AlgebraCore prefers

```python
a @ product @ b
```

instead of overloading

```python
a * b
```

as algebra multiplication.

This keeps different operations clearly separated:

- `scalar * element` means scalar scaling
- `a @ product @ b` means multiply with an explicit bilinear law
- `transformation @ element` means apply a linear map

The same `@` operator is therefore used consistently for contraction,
application, and composition of explicit algebraic structures.

## Design Notes

For a slightly more explicit discussion of the main API choices, see
`DESIGN.md`.

## Installation

For regular use, install directly from GitHub:

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

`AlgebraCore` is versioned and packaged as a normal Python project, but GitHub
is currently the intended installation path until the first PyPI release is
cut.

The release procedure for PyPI is described in `RELEASING.md`.

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
