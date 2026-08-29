from collections.abc import Sequence
from typing import final, override

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, Float, Real

from convax._utils import (
    as_float_array,
    normalize_affine_map_parameters,
    normalize_query_vector,
    require_finite,
    require_matrix,
)
from convax.sets._abstract import (
    AbstractAffineMapClosedSet,
    AbstractConvexHullClosedSet,
    AbstractSupportSet,
)
from convax.sets._results import SupportResult


@final
class VertexPolytope(
    AbstractAffineMapClosedSet,
    AbstractConvexHullClosedSet,
    AbstractSupportSet,
):
    """Convex hull of an explicit nonempty vertex collection.

    Args:
        vertices: Vertices with shape ``(vertex_count, ambient_dimension)``.

    Attributes:
        vertices: Vertices with shape ``(vertex_count, ambient_dimension)``.
        ambient_dimension: Dimension of the containing vector space.
        dtype: JAX dtype of the vertices.
    """

    vertices: Float[Array, "vertex_count ambient_dimension"]

    def __init__(
        self,
        vertices: Real[ArrayLike, "vertex_count ambient_dimension"]
        | Sequence[Sequence[float | int]],
    ) -> None:
        vertices = as_float_array(vertices)
        require_matrix("vertices", vertices)
        if vertices.shape[0] == 0:
            raise ValueError("vertices must contain at least one point")
        self.vertices = require_finite("vertices", vertices)

    @property
    def ambient_dimension(self) -> int:
        return self.vertices.shape[1]

    @property
    def dtype(self):
        return self.vertices.dtype

    def affine_map(
        self,
        matrix: Real[ArrayLike, "output_dimension {self.ambient_dimension}"]
        | Sequence[Sequence[float | int]],
        offset: Real[ArrayLike, "output_dimension"]
        | Sequence[float | int]
        | None = None,
    ) -> "VertexPolytope":
        """Return the affine image as a vertex polytope.

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
        return VertexPolytope(self.vertices.astype(matrix.dtype) @ matrix.T + offset)

    def convex_hull(self, other: AbstractConvexHullClosedSet) -> "VertexPolytope":
        """Return the convex hull as a vertex polytope.

        Args:
            other: Vertex polytope with the same ambient dimension.
        """
        if not isinstance(other, VertexPolytope):
            raise TypeError(
                "convex hull requires matching set types, got "
                f"VertexPolytope and {type(other).__name__}"
            )
        if self.ambient_dimension != other.ambient_dimension:
            raise ValueError(
                "convex hull dimensions must match, got "
                f"{self.ambient_dimension} and {other.ambient_dimension}"
            )
        dtype = jnp.result_type(self.dtype, other.dtype)
        return VertexPolytope(
            jnp.concatenate(
                (self.vertices.astype(dtype), other.vertices.astype(dtype)), axis=0
            )
        )

    @override
    def support(
        self,
        direction: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> SupportResult:
        """Return the support value and a maximizing vertex.

        Args:
            direction: Support-query vector with shape ``(ambient_dimension,)``.
        """
        direction = normalize_query_vector(
            "direction", direction, self.ambient_dimension, dtype=self.dtype
        )
        vertex_values = self.vertices @ direction
        maximizing_index = jnp.argmax(vertex_values)
        return SupportResult(
            value=vertex_values[maximizing_index],
            point=self.vertices[maximizing_index],
        )
