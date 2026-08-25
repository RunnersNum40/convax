from collections.abc import Sequence
from typing import final, override

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, Bool, Float, Real, ScalarLike

from convax._utils import (
    as_float_array,
    normalize_query_vector,
    normalize_tolerance,
    require_matrix,
    require_vector,
    require_vector_dimension,
)
from convax.sets._abstract import (
    AbstractIntersectionSet,
    AbstractNegationSet,
    AbstractPointContainmentSet,
    AbstractTranslationSet,
)


@final
class HalfspacePolyhedron(
    AbstractTranslationSet,
    AbstractNegationSet,
    AbstractPointContainmentSet,
    AbstractIntersectionSet,
):
    r"""Represents \(\{x \mid Ax \leq b, Ex = f\}\). The representation may
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
        inequality_matrix: Real[ArrayLike, "inequality_count ambient_dimension"]
        | Sequence[Sequence[float | int]],
        inequality_bounds: Real[ArrayLike, "inequality_count"] | Sequence[float | int],
        equality_matrix: Real[ArrayLike, "equality_count ambient_dimension"]
        | Sequence[Sequence[float | int]]
        | None = None,
        equality_values: Real[ArrayLike, "equality_count"]
        | Sequence[float | int]
        | None = None,
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

    def affine_preimage(
        self,
        matrix: Real[ArrayLike, "{self.ambient_dimension} input_dimension"]
        | Sequence[Sequence[float | int]],
        offset: Real[ArrayLike, "{self.ambient_dimension}"]
        | Sequence[float | int]
        | None = None,
    ) -> "HalfspacePolyhedron":
        r"""Return \(\{x \mid Ax + b \in P\}\).

        Args:
            matrix: Linear map with shape
                ``(ambient_dimension, input_dimension)``.
            offset: Optional vector with shape ``(ambient_dimension,)`` added
                before membership testing; ``None`` selects zero.
        """
        matrix = as_float_array(matrix)
        require_matrix("matrix", matrix)
        if matrix.shape[0] != self.ambient_dimension:
            raise ValueError(
                "matrix rows must match the polyhedron dimension, got "
                f"{matrix.shape} and {self.ambient_dimension}"
            )
        if offset is None:
            offset = jnp.zeros(matrix.shape[0], dtype=matrix.dtype)
        else:
            offset = as_float_array(offset)
        require_vector_dimension("offset", offset, matrix.shape[0])
        dtype = jnp.result_type(self.dtype, matrix.dtype, offset.dtype)
        matrix = matrix.astype(dtype)
        offset = offset.astype(dtype)
        inequality_matrix = self.inequality_matrix.astype(dtype)
        inequality_bounds = self.inequality_bounds.astype(dtype)
        equality_matrix = self.equality_matrix.astype(dtype)
        equality_values = self.equality_values.astype(dtype)
        return HalfspacePolyhedron(
            inequality_matrix @ matrix,
            inequality_bounds - inequality_matrix @ offset,
            equality_matrix @ matrix,
            equality_values - equality_matrix @ offset,
        )

    @override
    def translate(
        self,
        offset: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> "HalfspacePolyhedron":
        """Return the translated halfspace polyhedron.

        Args:
            offset: Translation vector with shape ``(ambient_dimension,)``.
        """
        offset = as_float_array(offset)
        require_vector_dimension("offset", offset, self.ambient_dimension)
        dtype = jnp.result_type(self.dtype, offset.dtype)
        inequality_matrix = self.inequality_matrix.astype(dtype)
        inequality_bounds = self.inequality_bounds.astype(dtype)
        equality_matrix = self.equality_matrix.astype(dtype)
        equality_values = self.equality_values.astype(dtype)
        offset = offset.astype(dtype)
        return HalfspacePolyhedron(
            inequality_matrix,
            inequality_bounds + inequality_matrix @ offset,
            equality_matrix,
            equality_values + equality_matrix @ offset,
        )

    @override
    def negate(self) -> "HalfspacePolyhedron":
        return HalfspacePolyhedron(
            -self.inequality_matrix,
            self.inequality_bounds,
            -self.equality_matrix,
            self.equality_values,
        )

    @override
    def intersection(self, other: AbstractIntersectionSet) -> "HalfspacePolyhedron":
        """Return the intersection as a halfspace polyhedron.

        Args:
            other: Halfspace polyhedron with the same ambient dimension.
        """
        if not isinstance(other, HalfspacePolyhedron):
            raise TypeError(
                "intersection requires matching representations, got "
                f"HalfspacePolyhedron and {type(other).__name__}"
            )
        if self.ambient_dimension != other.ambient_dimension:
            raise ValueError(
                "intersection dimensions must match, got "
                f"{self.ambient_dimension} and {other.ambient_dimension}"
            )
        return HalfspacePolyhedron(
            jnp.concatenate((self.inequality_matrix, other.inequality_matrix), axis=0),
            jnp.concatenate((self.inequality_bounds, other.inequality_bounds), axis=0),
            jnp.concatenate((self.equality_matrix, other.equality_matrix), axis=0),
            jnp.concatenate((self.equality_values, other.equality_values), axis=0),
        )

    @override
    def contains(
        self,
        point: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
        *,
        tolerance: ScalarLike = 1e-6,
    ) -> Bool[Array, ""]:
        """Return whether a point satisfies every constraint.

        Args:
            point: Query point with shape ``(ambient_dimension,)``.
            tolerance: Finite, nonnegative scalar feasibility tolerance.
        """
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
