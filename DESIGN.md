# Design Notes

This file records a few deliberate API decisions in `AlgebraCore`.

## 1. Why Is The Product Explicit?

`AlgebraCore` does not treat algebra multiplication as something hidden inside
the `Element` type.

Instead, the multiplication law is represented explicitly by an
`AlgebraProduct`. That means the same basis and the same coefficient vectors can
be combined with different bilinear products.

This is important because many familiar algebras differ mainly by their product
law:

- complex and dual numbers
- matrix and polynomial algebras
- Clifford, exterior, and interior products
- custom products defined by structure constants

Representing the product explicitly keeps that difference visible in code.

## 2. Why `a @ product @ b` Instead Of `a * b`?

The expression

```python
a @ product @ b
```

makes the product law part of the expression itself.

This has several advantages:

- it avoids overloading `*` with algebra multiplication
- it leaves `*` available for scalar scaling
- it works naturally with multiple products on the same basis
- it matches the rest of the API, where `@` already means contraction,
  application, or composition of explicit structures

So in `AlgebraCore`:

- `scalar * element` means scaling
- `a @ product @ b` means algebra multiplication with an explicit bilinear law
- `transformation @ element` means linear application

The intent is that `@` signals: "combine explicit algebraic tensors by
contraction."

## 3. What Does `TensorBasis` Mean?

`TensorBasis` is a computational helper name, not a claim that there is a
separate mathematical primitive beyond `Basis`.

It simply constructs the basis of a tensor product space by concatenating the
axes of factor bases. The result is still an ordinary `Basis`.

For example,

```python
TensorBasis([Basis(["a", "b"]), Basis(["x", "y"])])
```

produces the same kind of object as

```python
Basis([["a", "b"], ["x", "y"]])
```

The helper exists because that construction appears repeatedly in code:

- outer products of elements
- tensor products of transformations
- tensor products of algebra products

So the name is about programming clarity and reuse.

## 4. Why Are There Two Transformation Viewpoints?

`Transformation` appears in two closely related ways:

- `tf @ elem` treats the transformation as an explicit tensor that acts by
  contraction
- `elem.transform(tf)` treats it as a basis-change object and therefore uses
  the inverse matrix on coefficients

Both viewpoints are needed:

- one for computation with explicit multilinear arrays
- one for coordinate changes between basis descriptions

The API keeps both available, but they are intentionally not the same
operation.
