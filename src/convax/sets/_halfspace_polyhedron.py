from typing import final, override

import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, ScalarLike

from convax._arrays import as_float_array, require_matrix, require_vector
from convax._types import MatrixLike, VectorLike
from convax.sets._abstract import (
    AbstractPointContainmentSet,
    normalize_query_vector,
    normalize_tolerance,
)


@final
class HalfspacePolyhedron(AbstractPointContainmentSet):
    r"""A polyhedron defined by affine inequalities and equalities.

    Represents :math:`\{x \mid Ax \leq b, Ex = f\}`. The representation may
    describe an unbounded, lower-dimensional, or empty set.

    Args:
        inequality_matrix: Matrix ``A`` with shape
            ``(inequality_count, ambient_dimension)``.
        inequality_bounds: Vector ``b`` with shape ``(inequality_count,)``.
        equality_matrix: Optional matrix ``E`` with shape
            ``(equality_count, ambient_dimension)``.
        equality_values: Optional vector ``f`` with shape ``(equality_count,)``.
    """

    inequality_matrix: Float[Array, "inequality_count ambient_dimension"]
    inequality_bounds: Float[Array, "inequality_count"]
    equality_matrix: Float[Array, "equality_count ambient_dimension"]
    equality_values: Float[Array, "equality_count"]

    def __init__(
        self,
        inequality_matrix: MatrixLike,
        inequality_bounds: VectorLike,
        equality_matrix: MatrixLike | None = None,
        equality_values: VectorLike | None = None,
    ) -> None:
        inequality_matrix = as_float_array(inequality_matrix)
        inequality_bounds = as_float_array(inequality_bounds)
        require_matrix("inequality_matrix", inequality_matrix)
        require_vector("inequality_bounds", inequality_bounds)
        if inequality_matrix.shape[0] != inequality_bounds.shape[0]:
            raise ValueError(
                "inequality_matrix rows must match inequality_bounds, got "
                f"{inequality_matrix.shape} and {inequality_bounds.shape}"
            )
        if equality_matrix is None and equality_values is None:
            equality_matrix = jnp.empty(
                (0, inequality_matrix.shape[1]),
                dtype=inequality_matrix.dtype,
            )
            equality_values = jnp.empty((0,), dtype=inequality_bounds.dtype)
        elif equality_matrix is not None and equality_values is not None:
            equality_matrix = as_float_array(equality_matrix)
            equality_values = as_float_array(equality_values)
        else:
            raise ValueError(
                "equality_matrix and equality_values must be provided together"
            )
        require_matrix("equality_matrix", equality_matrix)
        require_vector("equality_values", equality_values)
        if equality_matrix.shape[1] != inequality_matrix.shape[1]:
            raise ValueError(
                "equality_matrix columns must match inequality_matrix columns, got "
                f"{equality_matrix.shape} and {inequality_matrix.shape}"
            )
        if equality_matrix.shape[0] != equality_values.shape[0]:
            raise ValueError(
                "equality_matrix rows must match equality_values, got "
                f"{equality_matrix.shape} and {equality_values.shape}"
            )
        dtype = jnp.result_type(
            inequality_matrix.dtype,
            inequality_bounds.dtype,
            equality_matrix.dtype,
            equality_values.dtype,
        )
        self.inequality_matrix = inequality_matrix.astype(dtype)
        self.inequality_bounds = inequality_bounds.astype(dtype)
        self.equality_matrix = equality_matrix.astype(dtype)
        self.equality_values = equality_values.astype(dtype)

    @property
    def ambient_dimension(self) -> int:
        return self.inequality_matrix.shape[1]

    @property
    def dtype(self):
        return self.inequality_matrix.dtype

    @override
    def contains(
        self,
        point: VectorLike,
        *,
        tolerance: ScalarLike = 1e-6,
    ) -> Bool[Array, ""]:
        point = normalize_query_vector(
            "point", point, self.ambient_dimension, dtype=self.dtype
        )
        tolerance = normalize_tolerance(tolerance, dtype=self.dtype)
        dtype = jnp.result_type(self.dtype, point.dtype, tolerance.dtype)
        point = point.astype(dtype)
        tolerance = tolerance.astype(dtype)
        inequality_matrix = self.inequality_matrix.astype(dtype)
        inequality_bounds = self.inequality_bounds.astype(dtype)
        equality_matrix = self.equality_matrix.astype(dtype)
        equality_values = self.equality_values.astype(dtype)
        satisfies_inequalities = jnp.all(
            inequality_matrix @ point <= inequality_bounds + tolerance
        )
        satisfies_equalities = jnp.all(
            jnp.abs(equality_matrix @ point - equality_values) <= tolerance
        )
        return jnp.logical_and(satisfies_inequalities, satisfies_equalities)
