# Design Notes

This file records a few deliberate API decisions in `AlgebraCore`.

## 1. Why Is The Product Explicit?

`AlgebraCore` does not treat algebra multiplication as something hidden inside
the `Element` type.

Instead, the multiplication law is represented explicitly by an
`AlgebraProduct`. That means the same basis and the same coefficient vectors can
be combined with different bilinear products.

This is a deliberate separation of concerns:

- a `Basis` describes labels and coordinates
- an `AlgebraProduct` describes one bilinear law on that space

So a basis is not tied one-to-one to a unique product. The same basis can carry
different bilinear laws, for example Clifford, exterior, and interior products
on the same underlying coordinate space.

This is important because many familiar algebras differ mainly by their product
law:

- complex and dual numbers
- matrix and polynomial algebras
- Clifford, exterior, and interior products
- custom products defined by structure constants

Representing the product explicitly keeps that difference visible in code.

## 1a. Why Does `AlgebraCore.std` Export Canonical Product Objects?

For the fixed examples in `AlgebraCore.std`, the preferred API is now a
canonical object such as `complex_product` or `so3_lie_bracket`, not only a
factory function.

That is only a convenience choice for the bundled showcase examples. It is not
a statement that a basis has one privileged product.

The point is:

- `complex_basis` and `complex_product` are convenient canonical representatives
- the design still allows many different products on the same basis
- the callable form is kept so the same canonical product can be rebuilt on
  another matching basis instance when needed

## 1b. Why Are Structure Constants A Core Feature?

One of the main points of `AlgebraCore` is that user-defined finite-dimensional
algebras are not an edge case. They are part of the central design.

If you can choose a basis and write down structure constants

```python
C[i, j, k]
```

for the multiplication law, then you can build the algebra directly.

That means the package is not only for the bundled examples in
`AlgebraCore.std`. Those examples are just convenient instances of the same
general representation.

The intended way to define such an algebra is plain Python and NumPy:

- define basis labels with `Basis(...)`
- fill a dense structure-constant tensor with NumPy
- construct `AlgebraProduct(basis, C)`

No additional configuration language is required.

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

## 5. Why Is This Different From Just Using NumPy?

`AlgebraCore` is intentionally close to NumPy, but it adds structure that plain
arrays do not carry by themselves.

NumPy gives efficient dense array operations. `AlgebraCore` keeps that dense
backend, but adds:

- named bases instead of anonymous axes
- explicit bilinear products instead of ad hoc multiplication code
- reusable transformation objects instead of one-off matrix manipulations
- a uniform representation for many different finite-dimensional algebras

So the goal is not to replace NumPy. The goal is to make algebraic structure
explicit while still staying in NumPy's fast dense world.
