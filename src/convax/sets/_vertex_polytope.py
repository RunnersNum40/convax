from typing import final, override

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from convax._arrays import as_float_array, require_matrix
from convax._types import MatrixLike, VectorLike
from convax.sets._abstract import AbstractSupportSet, normalize_query_vector
from convax.sets._results import SupportResult


@final
class VertexPolytope(AbstractSupportSet):
    """The convex hull of an explicit nonempty collection of vertices.

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
