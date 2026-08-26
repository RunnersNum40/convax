import jax.numpy as jnp
from jax import Array

from convax import operations, sets


class PointSet(sets.AbstractAffineMapClosedSet):
    point: Array

    def __init__(self, point):
        self.point = jnp.asarray(point)

    @property
    def ambient_dimension(self):
        return self.point.shape[0]

    @property
    def dtype(self):
        return self.point.dtype

    def affine_map(self, matrix, offset=None):
        matrix = jnp.asarray(matrix)
        if offset is None:
            offset = jnp.zeros(matrix.shape[0], dtype=matrix.dtype)
        else:
            offset = jnp.asarray(offset)
        return PointSet(matrix @ self.point + offset)


point_set = PointSet([1.0, 2.0])
mapped = operations.affine_map(point_set, [[2.0, 0.0]], [1.0])  # PointSet
