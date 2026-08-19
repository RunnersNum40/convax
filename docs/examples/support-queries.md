# Batched Support Queries

For a compact convex set \(X\), the support function in direction \(d\) is

\[
h_X(d) = \max_{x \in X} d^\mathsf{T}x.
\]

It gives a directional extremum and a boundary sample through a maximizing point. Convax implements the one-direction kernel; the application owns batching and compilation:

The example constructs a two-dimensional `Zonotope`, evaluates eight directions, and returns support values, maximizing points, and tight axis-aligned bounds from one compiled function. The JAX-array result feeds directly into a larger transformed program.

When a direction is orthogonal to a zonotope generator, the maximizing point is not unique: the support value remains exact, but gradients through the selected point are not meaningful at the tie.

<!-- fmt:off -->
```python title="examples/support_queries.py"
--8<-- "examples/support_queries.py"
```
<!-- fmt:on -->
