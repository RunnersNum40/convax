from collections.abc import Sequence
from typing import final, override

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, Float, Real

from convax._utils import (
    normalize_affine_map_parameters,
    normalize_query_vector,
)
from convax.sets._abstract import (
    AbstractAffineMapClosedSet,
    AbstractSupportSet,
)
from convax.sets._results import SupportResult


@final
class AffineImage(AbstractAffineMapClosedSet, AbstractSupportSet):
    r"""Support-set affine image \(\{Ax + b \mid x \in X\}\).

    Args:
        convex_set: Support-capable source set.
        matrix: Linear-map matrix with shape
            ``(output_dimension, convex_set.ambient_dimension)``.
        offset: Optional translation vector with shape ``(output_dimension,)``;
            ``None`` selects zero.
    """

    convex_set: AbstractSupportSet
    matrix: Float[Array, "output_dimension input_dimension"]
    offset: Float[Array, "output_dimension"]

    def __init__(
        self,
        convex_set: AbstractSupportSet,
        matrix: Real[ArrayLike, "output_dimension {convex_set.ambient_dimension}"]
        | Sequence[Sequence[float | int]],
        offset: Real[ArrayLike, "output_dimension"]
        | Sequence[float | int]
        | None = None,
    ) -> None:
        matrix, offset = normalize_affine_map_parameters(
            matrix,
            offset,
            convex_set.ambient_dimension,
            dtype=convex_set.dtype,
        )
        self.convex_set = convex_set
        self.matrix = matrix
        self.offset = offset

    @property
    def ambient_dimension(self) -> int:
        return self.matrix.shape[0]

    @property
    def dtype(self):
        return self.matrix.dtype

    def affine_map(
        self,
        matrix: Real[ArrayLike, "output_dimension {self.ambient_dimension}"]
        | Sequence[Sequence[float | int]],
        offset: Real[ArrayLike, "output_dimension"]
        | Sequence[float | int]
        | None = None,
    ) -> "AffineImage":
        """Return the composed affine image.

        Args:
            matrix: Linear-map matrix with shape
                ``(output_dimension, ambient_dimension)``.
            offset: Optional translation vector with shape ``(output_dimension,)``;
                ``None`` selects zero.
        """
        matrix, offset = normalize_affine_map_parameters(
            matrix,
            offset,
            self.ambient_dimension,
            dtype=self.dtype,
        )
        return AffineImage(
            self.convex_set,
            matrix @ self.matrix,
            matrix @ self.offset + offset,
        )

    @override
    def support(
        self,
        direction: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> SupportResult:
        """Return the support value and a maximizing point.

        Args:
            direction: Support-query vector with shape ``(ambient_dimension,)``.
        """
        direction = normalize_query_vector(
            "direction", direction, self.ambient_dimension, dtype=self.dtype
        )
        source_support = self.convex_set.support(self.matrix.T @ direction)
        return SupportResult(
            value=source_support.value + self.offset @ direction,
            point=self.matrix @ source_support.point + self.offset,
        )


@final
class MinkowskiSum(AbstractSupportSet):
    r"""Minkowski sum \(\{x + y \mid x \in X, y \in Y\}\).

    Args:
        left_set: First support-capable operand.
        right_set: Second support-capable operand with the same ambient dimension.
    """

    left_set: AbstractSupportSet
    right_set: AbstractSupportSet

    def __init__(
        self, left_set: AbstractSupportSet, right_set: AbstractSupportSet
    ) -> None:
        if left_set.ambient_dimension != right_set.ambient_dimension:
            raise ValueError(
                "Minkowski sum dimensions must match, got "
                f"{left_set.ambient_dimension} and {right_set.ambient_dimension}"
            )
        self.left_set = left_set
        self.right_set = right_set

    @property
    def ambient_dimension(self) -> int:
        return self.left_set.ambient_dimension

    @property
    def dtype(self):
        return jnp.result_type(self.left_set.dtype, self.right_set.dtype)

    @override
    def support(
        self,
        direction: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> SupportResult:
        """Return the support value and a maximizing point.

        Args:
            direction: Support-query vector with shape ``(ambient_dimension,)``.
        """
        direction = normalize_query_vector(
            "direction", direction, self.ambient_dimension, dtype=self.dtype
        )
        left_support = self.left_set.support(direction)
        right_support = self.right_set.support(direction)
        return SupportResult(
            value=left_support.value + right_support.value,
            point=left_support.point + right_support.point,
        )


@final
class ConvexHull(AbstractSupportSet):
    """Convex hull of two compact convex sets.

    Args:
        left_set: First support-capable operand.
        right_set: Second support-capable operand with the same ambient dimension.
    """

    left_set: AbstractSupportSet
    right_set: AbstractSupportSet

    def __init__(
        self, left_set: AbstractSupportSet, right_set: AbstractSupportSet
    ) -> None:
        if left_set.ambient_dimension != right_set.ambient_dimension:
            raise ValueError(
                "convex hull dimensions must match, got "
                f"{left_set.ambient_dimension} and {right_set.ambient_dimension}"
            )
        self.left_set = left_set
        self.right_set = right_set

    @property
    def ambient_dimension(self) -> int:
        return self.left_set.ambient_dimension

    @property
    def dtype(self):
        return jnp.result_type(self.left_set.dtype, self.right_set.dtype)

    @override
    def support(
        self,
        direction: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> SupportResult:
        """Return the support value and a maximizing point.

        Args:
            direction: Support-query vector with shape ``(ambient_dimension,)``.
        """
        direction = normalize_query_vector(
            "direction", direction, self.ambient_dimension, dtype=self.dtype
        )
        left_support = self.left_set.support(direction)
        right_support = self.right_set.support(direction)
        return jax.lax.cond(
            left_support.value >= right_support.value,
            lambda: left_support,
            lambda: right_support,
        )
