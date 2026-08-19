from abc import abstractmethod
from typing import Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import Bool, Float, ScalarLike

from convax._arrays import (
    as_float_array,
    require_matrix,
    require_scalar,
    require_vector,
    require_vector_dimension,
)
from convax._types import IntegerVectorLike, MatrixLike, VectorLike
from convax.sets._results import AxisAlignedBounds, SupportResult


class AbstractConvexSet(eqx.Module):
    ambient_dimension: eqx.AbstractVar[int]
    dtype: eqx.AbstractVar[DTypeLike]


class AbstractTranslationSet(AbstractConvexSet):
    """Interface for translation-closed representations."""

    @abstractmethod
    def translate(self, offset: VectorLike) -> Self:
        """Return the translated set in the same concrete type."""


class AbstractNegationSet(AbstractConvexSet):
    """Interface for negation-closed representations."""

    @abstractmethod
    def negate(self) -> Self:
        """Return the negated set in the same concrete type."""


class AbstractAffineMapSet(AbstractTranslationSet, AbstractNegationSet):
    """Interface for representations closed under arbitrary affine maps."""

    @abstractmethod
    def affine_map(
        self,
        matrix: MatrixLike,
        offset: VectorLike | None = None,
    ) -> Self:
        """Return the affine image in the same concrete type."""

    def project_coordinates(self, coordinates: IntegerVectorLike) -> Self:
        """Project onto an ordered collection of coordinate axes."""
        coordinates = jnp.asarray(coordinates)
        if coordinates.ndim != 1:
            raise ValueError(
                f"coordinates must be a vector, got shape {coordinates.shape}"
            )
        if not jnp.issubdtype(coordinates.dtype, jnp.integer):
            raise TypeError("coordinates must contain integers")
        coordinates = eqx.error_if(
            coordinates,
            jnp.any((coordinates < 0) | (coordinates >= self.ambient_dimension)),
            "coordinates must lie within the set's ambient dimension",
        )
        projection_matrix = jax.nn.one_hot(
            coordinates,
            self.ambient_dimension,
            dtype=self.dtype,
        )
        return self.affine_map(projection_matrix)

    @override
    def translate(self, offset: VectorLike) -> Self:
        return self.affine_map(
            jnp.eye(self.ambient_dimension, dtype=self.dtype),
            offset,
        )

    @override
    def negate(self) -> Self:
        return self.affine_map(-jnp.eye(self.ambient_dimension, dtype=self.dtype))


class AbstractSupportSet(AbstractConvexSet):
    """Interface for compact convex sets with a support function."""

    @abstractmethod
    def support(self, direction: VectorLike) -> SupportResult:
        """Return the support value and a maximizing point."""

    def support_value(self, direction: VectorLike) -> Float[Array, ""]:
        return self.support(direction).value

    def support_point(self, direction: VectorLike) -> Float[Array, "ambient_dimension"]:
        return self.support(direction).point

    def axis_aligned_bounds(self) -> AxisAlignedBounds:
        """Return tight axis-aligned bounds."""
        coordinate_directions = jnp.eye(self.ambient_dimension, dtype=self.dtype)
        upper = jax.vmap(self.support_value)(coordinate_directions)
        lower = -jax.vmap(self.support_value)(-coordinate_directions)
        return AxisAlignedBounds(lower=lower, upper=upper)


class AbstractPointContainmentSet(AbstractConvexSet):
    @abstractmethod
    def contains(
        self,
        point: VectorLike,
        *,
        tolerance: ScalarLike = 1e-6,
    ) -> Bool[Array, ""]:
        """Check point containment.

        Args:
            point: Query point of shape ``(ambient_dimension,)``.
            tolerance: Finite, nonnegative scalar feasibility tolerance.

        Raises:
            TypeError: If an input contains complex values.
            ValueError: If an input has invalid rank or dimension.
            EquinoxRuntimeError: If ``tolerance`` is negative or nonfinite.
        """


def normalize_affine_map_parameters(
    matrix: MatrixLike,
    offset: VectorLike | None,
    input_dimension: int,
    *,
    dtype: DTypeLike,
) -> tuple[
    Float[Array, "output_dimension input_dimension"],
    Float[Array, "output_dimension"],
]:
    matrix = as_float_array(matrix)
    require_matrix("matrix", matrix)
    if matrix.shape[1] != input_dimension:
        raise ValueError(
            "matrix columns must match the set dimension, got "
            f"{matrix.shape} and {input_dimension}"
        )
    if offset is None:
        offset = jnp.zeros(matrix.shape[0], dtype=matrix.dtype)
    else:
        offset = as_float_array(offset)
    require_vector_dimension("offset", offset, matrix.shape[0])
    dtype = jnp.result_type(dtype, matrix.dtype, offset.dtype)
    return matrix.astype(dtype), offset.astype(dtype)


def _affine_map_center_and_generator_matrix(
    center: Float[Array, "input_dimension"],
    generator_matrix: Float[Array, "input_dimension generator_dimension"],
    matrix: MatrixLike,
    offset: VectorLike | None,
    *,
    source_dtype: DTypeLike,
) -> tuple[
    Float[Array, "output_dimension"],
    Float[Array, "output_dimension generator_dimension"],
]:
    matrix, offset = normalize_affine_map_parameters(
        matrix,
        offset,
        center.shape[0],
        dtype=source_dtype,
    )
    center = center.astype(matrix.dtype)
    generator_matrix = generator_matrix.astype(matrix.dtype)
    return matrix @ center + offset, matrix @ generator_matrix


def normalize_center_and_generator_matrix(
    center: VectorLike,
    generator_matrix: MatrixLike,
) -> tuple[
    Float[Array, "ambient_dimension"],
    Float[Array, "ambient_dimension latent_dimension"],
]:
    center = as_float_array(center)
    generator_matrix = as_float_array(generator_matrix)
    require_vector("center", center)
    require_matrix("generator_matrix", generator_matrix)
    if generator_matrix.shape[0] != center.shape[0]:
        raise ValueError(
            "generator_matrix rows must match the center dimension, got "
            f"{generator_matrix.shape} and {center.shape}"
        )
    dtype = jnp.result_type(center.dtype, generator_matrix.dtype)
    return center.astype(dtype), generator_matrix.astype(dtype)


def normalize_query_vector(
    name: str,
    value: VectorLike,
    ambient_dimension: int,
    *,
    dtype: DTypeLike,
) -> Float[Array, "ambient_dimension"]:
    value = as_float_array(value)
    require_vector_dimension(name, value, ambient_dimension)
    dtype = jnp.result_type(dtype, value.dtype)
    return value.astype(dtype)


def normalize_tolerance(tolerance: ScalarLike, *, dtype: DTypeLike) -> Float[Array, ""]:
    tolerance = as_float_array(tolerance)
    require_scalar("tolerance", tolerance)
    tolerance = eqx.error_if(
        tolerance,
        (tolerance < 0) | ~jnp.isfinite(tolerance),
        "tolerance must be finite and nonnegative",
    )
    dtype = jnp.result_type(dtype, tolerance.dtype)
    return tolerance.astype(dtype)
