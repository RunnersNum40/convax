<h1 align='center'>Convax</h1>

Convax is an Equinox-based convex set manipulation library for JAX whose
numerical operations compose with `jit`, `vmap`, and, where mathematically
applicable, `grad`.

The solver-free core currently provides ellipsoids, zonotopes, constrained
zonotopes, vertex polytopes, halfspace polyhedra, affine maps, coordinate
projections, Minkowski sums, convex hulls, and exact intersections for
halfspace and constrained-zonotope representations.

```python
import jax
import jax.numpy as jnp

from convax import operations, sets

ellipsoid = sets.Ellipsoid(
    center=jnp.array([1.0, -1.0]),
    generator_matrix=jnp.array([[2.0, 0.0], [0.0, 1.0]]),
)
directions = jnp.eye(2)
translated = operations.translate(ellipsoid, jnp.array([0.5, 0.0]))
support_values = jax.jit(jax.vmap(translated.support_value))(directions)
```

Numerical queries are methods; set-producing operations are free functions such
as `affine_map`, `translate`, `minkowski_sum`, and `intersection`. They retain an
exact concrete representation when available, otherwise use an exact solver-free
composite, and fail when neither construction exists.

Documentation is available at <https://convax.tedpinkerton.ca/>.
