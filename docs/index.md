# Convax

Convax is a [JAX](https://jax.readthedocs.io/en/latest/)-based library for
convex analysis. It provides convex sets and operations compatible with JAX
transformations such as `jax.jit`, `jax.grad` (with a few exceptions), and
`jax.vmap`.

## Installation

```console
pip install convax
```

```console
uv add convax
```

## Basic Usage

```python
import jax
import jax.numpy as jnp

from convax import Ellipsoid, translate

ellipsoid = Ellipsoid(
    center=jnp.array([1.0, -1.0]),  # (1)!
    generator_matrix=jnp.array([[2.0, 0.0], [0.0, 1.0]]),
)
directions = jnp.eye(2)
translated = translate(ellipsoid, [0.5, 0.0])
support_values = jax.vmap(translated.support_value)(directions)
```

1. You may pass a JAX array, NumPy array, or plain Python sequence to Convax
   objects. Convax converts inputs to JAX arrays and returns JAX arrays.

## API Shape

Numerical queries are methods, such as `support`, `axis_aligned_bounds`, and
`contains`; set-producing operations are free functions, such as `affine_map`,
`translate`, `minkowski_sum`, and `intersection`.

Free operations preserve an exact concrete representation when the operands are
closed under the operation, otherwise use an exact solver-free composite when
available, and raise `TypeError` when neither construction exists.
Construct `AffineImage`, `MinkowskiSum`, or `ConvexHull` directly only when a
specific lazy representation is required.
