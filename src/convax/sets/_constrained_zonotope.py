from typing import final, override

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import block_diag as block_diagonal
from jaxtyping import Float

from convax._utils import (
    MatrixLike,
    VectorLike,
    _affine_map_center_and_generator_matrix,
    as_float_array,
    normalize_center_and_generator_matrix,
    require_matrix,
    require_vector,
)
from convax.sets._abstract import (
    AbstractAffineMapSet,
    AbstractIntersectionSet,
    AbstractMinkowskiSumSet,
)


@final
class ConstrainedZonotope(
    AbstractAffineMapSet,
    AbstractIntersectionSet,
    AbstractMinkowskiSumSet,
):
    r"""Equality-constrained affine image of the unit infinity-norm ball.

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

    @override
    def affine_map(
        self,
        matrix: MatrixLike,
        offset: VectorLike | None = None,
    ) -> "ConstrainedZonotope":
        center, generator_matrix = _affine_map_center_and_generator_matrix(
            self.center,
            self.generator_matrix,
            matrix,
            offset,
            source_dtype=self.dtype,
        )
        return ConstrainedZonotope(
            center,
            generator_matrix,
            self.constraint_matrix,
            self.constraint_values,
        )

    @override
    def minkowski_sum(self, other: AbstractMinkowskiSumSet) -> "ConstrainedZonotope":
        if not isinstance(other, ConstrainedZonotope):
            raise TypeError(
                "Minkowski sum requires matching representations, got "
                f"ConstrainedZonotope and {type(other).__name__}"
            )
        if self.ambient_dimension != other.ambient_dimension:
            raise ValueError(
                "Minkowski sum dimensions must match, got "
                f"{self.ambient_dimension} and {other.ambient_dimension}"
            )
        dtype = jnp.result_type(self.dtype, other.dtype)
        left_generators = self.generator_matrix.astype(dtype)
        right_generators = other.generator_matrix.astype(dtype)
        output_constraints = block_diagonal(
            self.constraint_matrix.astype(dtype),
            other.constraint_matrix.astype(dtype),
        )
        return ConstrainedZonotope(
            self.center.astype(dtype) + other.center.astype(dtype),
            jnp.concatenate((left_generators, right_generators), axis=1),
            output_constraints,
            jnp.concatenate(
                (
                    self.constraint_values.astype(dtype),
                    other.constraint_values.astype(dtype),
                )
            ),
        )

    @override
    def intersection(self, other: AbstractIntersectionSet) -> "ConstrainedZonotope":
        if not isinstance(other, ConstrainedZonotope):
            raise TypeError(
                "intersection requires matching representations, got "
                f"ConstrainedZonotope and {type(other).__name__}"
            )
        if self.ambient_dimension != other.ambient_dimension:
            raise ValueError(
                "intersection dimensions must match, got "
                f"{self.ambient_dimension} and {other.ambient_dimension}"
            )
        dtype = jnp.result_type(self.dtype, other.dtype)
        left_center = self.center.astype(dtype)
        right_center = other.center.astype(dtype)
        left_generators = self.generator_matrix.astype(dtype)
        right_generators = other.generator_matrix.astype(dtype)
        operand_constraints = block_diagonal(
            self.constraint_matrix.astype(dtype),
            other.constraint_matrix.astype(dtype),
        )
        matching_constraints = jnp.concatenate(
            (left_generators, -right_generators), axis=1
        )
        return ConstrainedZonotope(
            left_center,
            jnp.concatenate(
                (left_generators, jnp.zeros_like(right_generators)), axis=1
            ),
            jnp.concatenate((operand_constraints, matching_constraints), axis=0),
            jnp.concatenate(
                (
                    self.constraint_values.astype(dtype),
                    other.constraint_values.astype(dtype),
                    right_center - left_center,
                )
            ),
        )
