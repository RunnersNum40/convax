from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import ArrayLike, Float, ScalarLike

type VectorLike = ArrayLike | Sequence[float | int]
type MatrixLike = ArrayLike | Sequence[Sequence[float | int]]
type IntegerVectorLike = ArrayLike | Sequence[int]


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


def normalize_affine_map_parameters(
    matrix: MatrixLike,
    offset: VectorLike | None,
    input_dimension: int,
    *,
    dtype: DTypeLike,
) -> tuple[
    Float[Array, "output_dimension input_dimension"],
    Float[Array, "output_dimension"],
]:
    matrix = as_float_array(matrix)
    require_matrix("matrix", matrix)
    if matrix.shape[1] != input_dimension:
        raise ValueError(
            "matrix columns must match the set dimension, got "
            f"{matrix.shape} and {input_dimension}"
        )
    if offset is None:
        offset = jnp.zeros(matrix.shape[0], dtype=matrix.dtype)
    else:
        offset = as_float_array(offset)
    require_vector_dimension("offset", offset, matrix.shape[0])
    dtype = jnp.result_type(dtype, matrix.dtype, offset.dtype)
    return matrix.astype(dtype), offset.astype(dtype)


def _affine_map_center_and_generator_matrix(
    center: Float[Array, "input_dimension"],
    generator_matrix: Float[Array, "input_dimension generator_dimension"],
    matrix: MatrixLike,
    offset: VectorLike | None,
    *,
    source_dtype: DTypeLike,
) -> tuple[
    Float[Array, "output_dimension"],
    Float[Array, "output_dimension generator_dimension"],
]:
    matrix, offset = normalize_affine_map_parameters(
        matrix,
        offset,
        center.shape[0],
        dtype=source_dtype,
    )
    center = center.astype(matrix.dtype)
    generator_matrix = generator_matrix.astype(matrix.dtype)
    return matrix @ center + offset, matrix @ generator_matrix


def normalize_center_and_generator_matrix(
    center: VectorLike,
    generator_matrix: MatrixLike,
) -> tuple[
    Float[Array, "ambient_dimension"],
    Float[Array, "ambient_dimension latent_dimension"],
]:
    center = as_float_array(center)
    generator_matrix = as_float_array(generator_matrix)
    require_vector("center", center)
    require_matrix("generator_matrix", generator_matrix)
    if generator_matrix.shape[0] != center.shape[0]:
        raise ValueError(
            "generator_matrix rows must match the center dimension, got "
            f"{generator_matrix.shape} and {center.shape}"
        )
    dtype = jnp.result_type(center.dtype, generator_matrix.dtype)
    return center.astype(dtype), generator_matrix.astype(dtype)


def normalize_query_vector(
    name: str,
    value: VectorLike,
    ambient_dimension: int,
    *,
    dtype: DTypeLike,
) -> Float[Array, "ambient_dimension"]:
    value = as_float_array(value)
    require_vector_dimension(name, value, ambient_dimension)
    dtype = jnp.result_type(dtype, value.dtype)
    return value.astype(dtype)


def normalize_tolerance(tolerance: ScalarLike, *, dtype: DTypeLike) -> Float[Array, ""]:
    tolerance = as_float_array(tolerance)
    require_scalar("tolerance", tolerance)
    tolerance = eqx.error_if(
        tolerance,
        (tolerance < 0) | ~jnp.isfinite(tolerance),
        "tolerance must be finite and nonnegative",
    )
    dtype = jnp.result_type(dtype, tolerance.dtype)
    return tolerance.astype(dtype)


def _scaled_l2_norm(vector: Float[Array, "dimension"]) -> Float[Array, ""]:
    output_dtype = vector.dtype
    if vector.shape[0] == 0:
        return jnp.zeros((), dtype=output_dtype)

    accumulation_dtype = jnp.result_type(output_dtype, jnp.float32)
    vector = vector.astype(accumulation_dtype)
    scale = jnp.max(jnp.abs(vector))
    norm = jax.lax.cond(
        jnp.isfinite(scale) & (scale > 0),
        lambda: scale * jnp.sqrt(jnp.sum(jnp.square(vector / scale))),
        lambda: scale,
    )
    return norm.astype(output_dtype)
