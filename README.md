<h1 align='center'>Convax</h1>

Convax is an Equinox-based JAX library for convex sets whose numerical
operations compose with `jit`, `vmap`, and, where applicable, `grad`.

Documentation is available at <https://convax.tedpinkerton.ca/>.

## Basic Example

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
