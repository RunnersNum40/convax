from typing import final

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from convax._arrays import as_float_array, require_matrix, require_vector
from convax._types import MatrixLike, VectorLike
from convax.sets._abstract import (
    AbstractConvexSet,
    normalize_center_and_generator_matrix,
)


@final
class ConstrainedZonotope(AbstractConvexSet):
    r"""An equality-constrained affine image of the unit infinity-norm ball.

    Represents
    :math:`\{c + G\xi \mid \lVert \xi \rVert_\infty \leq 1, A\xi = b\}`.
    Constraints may be redundant or infeasible; the conventional ``center`` is
    an affine offset that need not belong to the represented set.

    Args:
        center: Affine offset with shape ``(ambient_dimension,)``.
        generator_matrix: Generator matrix with shape
            ``(ambient_dimension, generator_count)``.
        constraint_matrix: Latent equality matrix with shape
            ``(constraint_count, generator_count)``.
        constraint_values: Latent equality values with shape
            ``(constraint_count,)``.
    """

    center: Float[Array, "ambient_dimension"]
    generator_matrix: Float[Array, "ambient_dimension generator_count"]
    constraint_matrix: Float[Array, "constraint_count generator_count"]
    constraint_values: Float[Array, "constraint_count"]

    def __init__(
        self,
        center: VectorLike,
        generator_matrix: MatrixLike,
        constraint_matrix: MatrixLike,
        constraint_values: VectorLike,
    ) -> None:
        center, generator_matrix = normalize_center_and_generator_matrix(
            center, generator_matrix
        )
        constraint_matrix = as_float_array(constraint_matrix)
        constraint_values = as_float_array(constraint_values)
        require_matrix("constraint_matrix", constraint_matrix)
        require_vector("constraint_values", constraint_values)
        if constraint_matrix.shape[1] != generator_matrix.shape[1]:
            raise ValueError(
                "constraint_matrix columns must match generator_matrix columns, got "
                f"{constraint_matrix.shape} and {generator_matrix.shape}"
            )
        if constraint_matrix.shape[0] != constraint_values.shape[0]:
            raise ValueError(
                "constraint_matrix rows must match constraint_values, got "
                f"{constraint_matrix.shape} and {constraint_values.shape}"
            )
        dtype = jnp.result_type(
            center.dtype,
            generator_matrix.dtype,
            constraint_matrix.dtype,
            constraint_values.dtype,
        )
        self.center = center.astype(dtype)
        self.generator_matrix = generator_matrix.astype(dtype)
        self.constraint_matrix = constraint_matrix.astype(dtype)
        self.constraint_values = constraint_values.astype(dtype)

    @property
    def ambient_dimension(self) -> int:
        return self.center.shape[0]

    @property
    def dtype(self):
        return self.center.dtype
