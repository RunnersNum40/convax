# Convax

Convax is a [JAX](https://jax.readthedocs.io/en/latest/)-based library for convex analysis.
It provides convex sets and operations on them compatible with JAX transformations such as `jax.jit`, `jax.grad` (where possible), and `jax.vmap`.

## Installation

```console
pip install convax
```

## Basic Usage

```python
import jax
import jax.numpy as jnp

from convax import Ellipsoid

ellipsoid = Ellipsoid(
    center=jnp.array([1.0, -1.0]),
    generator_matrix=jnp.array([[2.0, 0.0], [0.0, 1.0]]),
)
directions = jnp.eye(2)
support_values = jax.jit(jax.vmap(ellipsoid.support_value))(directions)
```
