from abc import abstractmethod

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
from convax._types import MatrixLike, VectorLike
from convax.sets._results import AxisAlignedBounds, SupportResult


class AbstractConvexSet(eqx.Module):
    """Interface shared by every Convax convex set."""

    ambient_dimension: eqx.AbstractVar[int]
    dtype: eqx.AbstractVar[DTypeLike]


class AbstractSupportSet(AbstractConvexSet):
    """Interface for compact convex sets with a support oracle."""

    @abstractmethod
    def support(self, direction: VectorLike) -> SupportResult:
        """Evaluate the support function and a maximizing point."""

    def support_value(self, direction: VectorLike) -> Float[Array, ""]:
        """Evaluate the support function in one direction."""
        return self.support(direction).value

    def support_point(self, direction: VectorLike) -> Float[Array, "ambient_dimension"]:
        """Return a maximizing point in one direction."""
        return self.support(direction).point

    def axis_aligned_bounds(self) -> AxisAlignedBounds:
        """Return the tight axis-aligned bounding box."""
        coordinate_directions = jnp.eye(self.ambient_dimension, dtype=self.dtype)
        upper = jax.vmap(self.support_value)(coordinate_directions)
        lower = -jax.vmap(self.support_value)(-coordinate_directions)
        return AxisAlignedBounds(lower=lower, upper=upper)


class AbstractPointContainmentSet(AbstractConvexSet):
    """Interface for sets with a direct point-containment predicate."""

    @abstractmethod
    def contains(
        self,
        point: VectorLike,
        *,
        tolerance: ScalarLike = 1e-6,
    ) -> Bool[Array, ""]:
        """Return whether one point lies in the set.

        Args:
            point: Query point with shape ``(ambient_dimension,)``.
            tolerance: Finite nonnegative scalar feasibility tolerance.

        Raises:
            TypeError: If an input contains complex values.
            ValueError: If an input has invalid rank or dimension.
            EquinoxRuntimeError: If ``tolerance`` is negative or non-finite.
        """


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
