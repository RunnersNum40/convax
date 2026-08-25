from abc import abstractmethod
from collections.abc import Sequence
from typing import Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import ArrayLike, Bool, Float, Real, ScalarLike

from convax.sets._results import AxisAlignedBounds, SupportResult


class AbstractConvexSet(eqx.Module):
    ambient_dimension: eqx.AbstractVar[int]
    dtype: eqx.AbstractVar[DTypeLike]


class AbstractTranslationSet(AbstractConvexSet):
    @abstractmethod
    def _translate(
        self,
        offset: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> Self:
        """Return the translated set in the concrete type."""


class AbstractNegationSet(AbstractConvexSet):
    @abstractmethod
    def _negate(self) -> Self:
        """Return the negated set in the concrete type."""


class AbstractAffineMapSet(AbstractTranslationSet, AbstractNegationSet):
    @abstractmethod
    def _affine_map(
        self,
        matrix: Real[ArrayLike, "output_dimension {self.ambient_dimension}"]
        | Sequence[Sequence[float | int]],
        offset: Real[ArrayLike, "output_dimension"]
        | Sequence[float | int]
        | None = None,
    ) -> Self:
        """Return the affine image in the concrete type."""

    @override
    def _translate(
        self,
        offset: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> Self:
        return self._affine_map(
            jnp.eye(self.ambient_dimension, dtype=self.dtype),
            offset,
        )

    @override
    def _negate(self) -> Self:
        return self._affine_map(-jnp.eye(self.ambient_dimension, dtype=self.dtype))


class AbstractAffinePreimageSet(AbstractConvexSet):
    @abstractmethod
    def _affine_preimage(
        self,
        matrix: Real[ArrayLike, "{self.ambient_dimension} input_dimension"]
        | Sequence[Sequence[float | int]],
        offset: Real[ArrayLike, "{self.ambient_dimension}"]
        | Sequence[float | int]
        | None = None,
    ) -> Self:
        """Return the affine preimage in the concrete type."""


class AbstractMinkowskiSumSet(AbstractConvexSet):
    @abstractmethod
    def _minkowski_sum(self, other: "AbstractMinkowskiSumSet") -> Self:
        """Return the Minkowski sum in its concrete type."""


class AbstractIntersectionSet(AbstractConvexSet):
    @abstractmethod
    def _intersection(self, other: "AbstractIntersectionSet") -> Self:
        """Return the intersection in its concrete type."""


class AbstractConvexHullSet(AbstractConvexSet):
    @abstractmethod
    def _convex_hull(self, other: "AbstractConvexHullSet") -> Self:
        """Return the convex hull in its concrete type."""


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
        """Return tight bounds for each coordinate."""
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
