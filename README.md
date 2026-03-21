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
