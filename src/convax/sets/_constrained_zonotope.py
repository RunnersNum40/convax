from collections.abc import Sequence
from typing import final

import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import block_diag as block_diagonal
from jaxtyping import ArrayLike, Float, Real

from convax._utils import (
    _affine_map_center_and_generator_matrix,
    as_float_array,
    normalize_center_and_generator_matrix,
    require_matrix,
    require_vector,
)
from convax.sets._abstract import (
    AbstractAffineMapClosedSet,
    AbstractIntersectionClosedSet,
    AbstractMinkowskiSumClosedSet,
)


@final
class ConstrainedZonotope(
    AbstractAffineMapClosedSet,
    AbstractIntersectionClosedSet,
    AbstractMinkowskiSumClosedSet,
):
    r"""Equality-constrained affine image of the unit infinity-norm ball.

    Represents
    \(\{c + G\xi \mid \lVert \xi \rVert_\infty \leq 1, A\xi = b\}\).
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

    Attributes:
        center: Affine offset with shape ``(ambient_dimension,)``.
        generator_matrix: Generator matrix with shape
            ``(ambient_dimension, generator_count)``.
        constraint_matrix: Latent equality matrix with shape
            ``(constraint_count, generator_count)``.
        constraint_values: Latent equality values with shape
            ``(constraint_count,)``.
        ambient_dimension: Dimension of the containing vector space.
        dtype: Common JAX dtype of the set arrays.
    """

    center: Float[Array, "ambient_dimension"]
    generator_matrix: Float[Array, "ambient_dimension generator_count"]
    constraint_matrix: Float[Array, "constraint_count generator_count"]
    constraint_values: Float[Array, "constraint_count"]

    def __init__(
        self,
        center: Real[ArrayLike, "ambient_dimension"] | Sequence[float | int],
        generator_matrix: Real[ArrayLike, "ambient_dimension generator_count"]
        | Sequence[Sequence[float | int]],
        constraint_matrix: Real[ArrayLike, "constraint_count generator_count"]
        | Sequence[Sequence[float | int]],
        constraint_values: Real[ArrayLike, "constraint_count"] | Sequence[float | int],
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

    def affine_map(
        self,
        matrix: Real[ArrayLike, "output_dimension {self.ambient_dimension}"]
        | Sequence[Sequence[float | int]],
        offset: Real[ArrayLike, "output_dimension"]
        | Sequence[float | int]
        | None = None,
    ) -> "ConstrainedZonotope":
        """Return the affine image as a constrained zonotope.

        Args:
            matrix: Linear-map matrix with shape
                ``(output_dimension, ambient_dimension)``.
            offset: Optional translation vector with shape ``(output_dimension,)``;
                ``None`` selects zero.
        """
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

    def minkowski_sum(
        self, other: AbstractMinkowskiSumClosedSet
    ) -> "ConstrainedZonotope":
        """Return the Minkowski sum as a constrained zonotope.

        Args:
            other: Constrained zonotope with the same ambient dimension.
        """
        if not isinstance(other, ConstrainedZonotope):
            raise TypeError(
                "Minkowski sum requires matching set types, got "
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

    def intersection(
        self, other: AbstractIntersectionClosedSet
    ) -> "ConstrainedZonotope":
        """Return the intersection as a constrained zonotope.

        Args:
            other: Constrained zonotope with the same ambient dimension.
        """
        if not isinstance(other, ConstrainedZonotope):
            raise TypeError(
                "intersection requires matching set types, got "
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
