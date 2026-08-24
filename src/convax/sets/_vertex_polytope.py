from typing import final, override

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from convax._utils import (
    MatrixLike,
    VectorLike,
    as_float_array,
    normalize_affine_map_parameters,
    normalize_query_vector,
    require_matrix,
)
from convax.sets._abstract import (
    AbstractAffineMapSet,
    AbstractConvexHullSet,
    AbstractSupportSet,
)
from convax.sets._results import SupportResult


@final
class VertexPolytope(
    AbstractAffineMapSet,
    AbstractConvexHullSet,
    AbstractSupportSet,
):
    """Convex hull of an explicit nonempty vertex collection.

    Args:
        vertices: Vertices with shape ``(vertex_count, ambient_dimension)``.
    """

    vertices: Float[Array, "vertex_count ambient_dimension"]

    def __init__(self, vertices: MatrixLike) -> None:
        vertices = as_float_array(vertices)
        require_matrix("vertices", vertices)
        if vertices.shape[0] == 0:
            raise ValueError("vertices must contain at least one point")
        self.vertices = vertices

    @property
    def ambient_dimension(self) -> int:
        return self.vertices.shape[1]

    @property
    def dtype(self):
        return self.vertices.dtype

    @override
    def affine_map(
        self,
        matrix: MatrixLike,
        offset: VectorLike | None = None,
    ) -> "VertexPolytope":
        matrix, offset = normalize_affine_map_parameters(
            matrix,
            offset,
            self.ambient_dimension,
            dtype=self.dtype,
        )
        return VertexPolytope(self.vertices.astype(matrix.dtype) @ matrix.T + offset)

    @override
    def convex_hull(self, other: AbstractConvexHullSet) -> "VertexPolytope":
        if not isinstance(other, VertexPolytope):
            raise TypeError(
                "convex hull requires matching representations, got "
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
    def support(self, direction: VectorLike) -> SupportResult:
        direction = normalize_query_vector(
            "direction", direction, self.ambient_dimension, dtype=self.dtype
        )
        vertex_values = self.vertices @ direction
        maximizing_index = jnp.argmax(vertex_values)
        return SupportResult(
            value=vertex_values[maximizing_index],
            point=self.vertices[maximizing_index],
        )
