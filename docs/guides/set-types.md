# Set Types and Capabilities

Convax distinguishes operations from closure of a concrete set type under them.
Generic functions in `convax.operations` use an exact construction when
available; a same-named method exists only when the result retains the concrete
set type.

For example, an affine image of an ellipsoid is another ellipsoid:

```python
mapped = ellipsoid.affine_map(matrix)
```

Both the method and `operations.affine_map(ellipsoid, matrix)` return an
`Ellipsoid`. Because two ellipsoids generally sum to a non-ellipsoid,
`Ellipsoid` has no `minkowski_sum` method; the generic operation returns the
exact `MinkowskiSum` composite:

```python
summed = operations.minkowski_sum(left_ellipsoid, right_ellipsoid)
```

## Capability Matrix

| Set type | Closed-operation methods | Query methods |
| --- | --- | --- |
| `Zonotope` | `affine_map`, `translate`, `negate` | `support`, `axis_aligned_bounds` |
| `Ellipsoid` | `affine_map`, `translate`, `negate` | `support`, `axis_aligned_bounds`, `contains` |
| `VertexPolytope` | `affine_map`, `translate`, `negate`, `convex_hull` | `support`, `axis_aligned_bounds` |
| `ConstrainedZonotope` | `affine_map`, `translate`, `negate`, `minkowski_sum`, `intersection` | - |
| `HalfspacePolyhedron` | `affine_preimage`, `translate`, `negate`, `intersection` | `contains` |
| `AffineImage` | `affine_map`, `translate`, `negate` | `support`, `axis_aligned_bounds` |
| `MinkowskiSum` | - | `support`, `axis_aligned_bounds` |
| `ConvexHull` | - | `support`, `axis_aligned_bounds` |

`support_value` and `support_point` are also available on every support-capable
set type.

## Generic Dispatch

Generic operations use these exact constructions:

| Operation | Type-preserving path | Composite fallback |
| --- | --- | --- |
| `affine_map` | `AbstractAffineMapClosedSet.affine_map` | `AffineImage` for support-capable sets |
| `project_coordinates` | Derived from `affine_map` | `AffineImage` for support-capable sets |
| `translate` | `AbstractTranslationClosedSet.translate` | `AffineImage` for support-capable sets |
| `negate` | `AbstractNegationClosedSet.negate` | `AffineImage` for support-capable sets |
| `minkowski_sum` | Matching `AbstractMinkowskiSumClosedSet` types | `MinkowskiSum` for support-capable sets |
| `convex_hull` | Matching `AbstractConvexHullClosedSet` types | `ConvexHull` for support-capable sets |
| `affine_preimage` | `AbstractAffinePreimageClosedSet.affine_preimage` | None |
| `intersection` | Matching `AbstractIntersectionClosedSet` types | None |

Binary type-preserving methods require the same concrete type and ambient
dimension; composite results retain only their implemented capabilities.

See the [set interfaces](../api/sets/abstract.md) for complete signatures and
[Extending Convax](extending.md) for the implementation contract.
