# Convax

Convax is a [JAX](https://jax.readthedocs.io/en/latest/)-based convex-analysis
library providing sets and operations compatible with `jax.jit`, `jax.grad`
(with exceptions), and `jax.vmap`.

## Installation

```console
pip install convax
```
or
```console
uv add convax
```

## Basic Usage

```python
import jax
import jax.numpy as jnp

from convax import operations, sets

ellipsoid = sets.Ellipsoid(
    center=jnp.array([1.0, -1.0]),  # (1)!
    generator_matrix=jnp.array([[2.0, 0.0], [0.0, 1.0]]),
)
directions = jnp.eye(2)
translated = operations.translate(ellipsoid, [0.5, 0.0])
support_values = jax.vmap(translated.support_value)(directions)
```

1. Convax accepts JAX arrays, NumPy arrays, and plain Python sequences,
   converting inputs and outputs to JAX arrays.

See [Set Types and Capabilities](guides/set-types.md) to choose a set type and
understand return types, or [Extending Convax](guides/extending.md) to implement
a custom set type.

## Credits

Shout out to [Equinox](https://docs.kidger.site/equinox/) for the awesome
library that Convax is build on. Also shout out to
[distreqx](https://lockwo.github.io/distreqx/) and
[Diffrax](https://docs.kidger.site/diffrax/) for the Equinox-based math-library
inspiration.
