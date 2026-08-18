import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from convax._types import MatrixLike, VectorLike


def as_float_array(value: VectorLike | MatrixLike) -> Float[Array, "..."]:
    value = jnp.asarray(value)
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise TypeError("Convax requires real-valued arrays")
    if not jnp.issubdtype(value.dtype, jnp.floating):
        value = value.astype(jnp.result_type(0.0))
    return value


def require_matrix(name: str, array: Array) -> None:
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got shape {array.shape}")


def require_scalar(name: str, array: Array) -> None:
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")


def require_vector(name: str, array: Array) -> None:
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector, got shape {array.shape}")


def require_vector_dimension(name: str, array: Array, dimension: int) -> None:
    require_vector(name, array)
    if array.shape[0] != dimension:
        raise ValueError(
            f"{name} must have dimension {dimension}, got shape {array.shape}"
        )
