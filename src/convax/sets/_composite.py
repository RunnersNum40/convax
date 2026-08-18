from typing import final, override

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from convax._arrays import as_float_array, require_matrix, require_vector_dimension
from convax._types import MatrixLike, VectorLike
from convax.sets._abstract import AbstractSupportSet, normalize_query_vector
from convax.sets._results import SupportResult


@final
class AffineImage(AbstractSupportSet):
    r"""An affine image :math:`\{Ax + b \mid x \in X\}` of a support set."""

    convex_set: AbstractSupportSet
    matrix: Float[Array, "output_dimension input_dimension"]
    offset: Float[Array, "output_dimension"]

    def __init__(
        self,
        convex_set: AbstractSupportSet,
        matrix: MatrixLike,
        offset: VectorLike | None = None,
    ) -> None:
        matrix = as_float_array(matrix)
        require_matrix("matrix", matrix)
        if matrix.shape[1] != convex_set.ambient_dimension:
            raise ValueError(
                "matrix columns must match the set dimension, got "
                f"{matrix.shape} and {convex_set.ambient_dimension}"
            )
        if offset is None:
            offset = jnp.zeros(matrix.shape[0], dtype=matrix.dtype)
        else:
            offset = as_float_array(offset)
        require_vector_dimension("offset", offset, matrix.shape[0])
        dtype = jnp.result_type(convex_set.dtype, matrix.dtype, offset.dtype)
        self.convex_set = convex_set
        self.matrix = matrix.astype(dtype)
        self.offset = offset.astype(dtype)

    @property
    def ambient_dimension(self) -> int:
        return self.matrix.shape[0]

    @property
    def dtype(self):
        return self.matrix.dtype

    @override
    def support(self, direction: VectorLike) -> SupportResult:
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
    r"""The Minkowski sum :math:`\{x + y \mid x \in X, y \in Y\}`."""

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
    def support(self, direction: VectorLike) -> SupportResult:
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
    """The convex hull of two compact convex sets."""

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
    def support(self, direction: VectorLike) -> SupportResult:
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
