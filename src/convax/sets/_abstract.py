from abc import abstractmethod
from collections.abc import Sequence
from typing import Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import ArrayLike, Bool, Float, Integer, Real, ScalarLike

from convax.sets._results import AxisAlignedBounds, SupportResult


class AbstractConvexSet(eqx.Module):
    ambient_dimension: eqx.AbstractVar[int]
    dtype: eqx.AbstractVar[DTypeLike]


class AbstractTranslationSet(AbstractConvexSet):
    """Interface for translation-closed representations."""

    @abstractmethod
    def translate(
        self,
        offset: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> Self:
        """Return the translated set in the same concrete type.

        Args:
            offset: Translation vector with shape ``(ambient_dimension,)``.
        """


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
        matrix: Real[ArrayLike, "output_dimension {self.ambient_dimension}"]
        | Sequence[Sequence[float | int]],
        offset: Real[ArrayLike, "output_dimension"]
        | Sequence[float | int]
        | None = None,
    ) -> Self:
        """Return the affine image in the same concrete type.

        Args:
            matrix: Linear-map matrix with shape
                ``(output_dimension, ambient_dimension)``.
            offset: Optional translation vector with shape
                ``(output_dimension,)``. ``None`` selects a zero offset.
        """

    def project_coordinates(
        self,
        coordinates: Integer[ArrayLike, "output_dimension"] | Sequence[int],
    ) -> Self:
        """Project onto an ordered collection of coordinate axes.

        Args:
            coordinates: One-dimensional integer sequence of coordinate indices.
                Its order determines the output coordinate order.
        """
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
    def translate(
        self,
        offset: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> Self:
        """Return the translated set.

        Args:
            offset: Translation vector with shape ``(ambient_dimension,)``.
        """
        return self.affine_map(
            jnp.eye(self.ambient_dimension, dtype=self.dtype),
            offset,
        )

    @override
    def negate(self) -> Self:
        return self.affine_map(-jnp.eye(self.ambient_dimension, dtype=self.dtype))


class AbstractMinkowskiSumSet(AbstractConvexSet):
    """Representation closed under Minkowski addition."""

    @abstractmethod
    def minkowski_sum(self, other: "AbstractMinkowskiSumSet") -> Self:
        """Return the Minkowski sum in its concrete type.

        Args:
            other: Set of the same concrete representation and ambient dimension.
        """


class AbstractIntersectionSet(AbstractConvexSet):
    """Representation closed under intersection."""

    @abstractmethod
    def intersection(self, other: "AbstractIntersectionSet") -> Self:
        """Return the intersection in its concrete type.

        Args:
            other: Set of the same concrete representation and ambient dimension.
        """


class AbstractConvexHullSet(AbstractConvexSet):
    """Representation closed under convex hull."""

    @abstractmethod
    def convex_hull(self, other: "AbstractConvexHullSet") -> Self:
        """Return the convex hull in its concrete type.

        Args:
            other: Set of the same concrete representation and ambient dimension.
        """


class AbstractSupportSet(AbstractConvexSet):
    """Interface for compact convex sets with a support function."""

    @abstractmethod
    def support(
        self,
        direction: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> SupportResult:
        """Return the support value and a maximizing point.

        Args:
            direction: Support-query vector with shape ``(ambient_dimension,)``.
        """

    def support_value(
        self,
        direction: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> Float[Array, ""]:
        """Return the support value.

        Args:
            direction: Support-query vector with shape ``(ambient_dimension,)``.
        """
        return self.support(direction).value

    def support_point(
        self,
        direction: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> Float[Array, "{self.ambient_dimension}"]:
        """Return a maximizing support point.

        Args:
            direction: Support-query vector with shape ``(ambient_dimension,)``.
        """
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
        point: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
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
