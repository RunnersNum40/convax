# Extending Convax

Custom set types are Equinox modules implementing `AbstractConvexSet` through
one or more nominal capability interfaces.

Custom set types expose `ambient_dimension` and `dtype` properties. For JIT
compatibility, use JAX arrays or valid JAX PyTrees for non-static fields, avoid
Python control flow over traced values, and normalize array-like inputs while
rejecting invalid ranks and dimensions at API boundaries.

## Closure Methods

Implement a closure interface only when its method returns the same concrete
type exactly. Although direct calls such as `affine_map` are supported, prefer
`convax.operations`, which can also select exact composite set types.

The following custom point-set type is affine-map closed and therefore inherits
type-preserving `translate` and `negate` methods:

## Custom Affine Closed Set

<!-- fmt:off -->
```python title="examples/custom_set.py"
--8<-- "examples/custom_set.py"
```
<!-- fmt:on -->
