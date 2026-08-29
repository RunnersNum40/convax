from abc import abstractmethod
from collections.abc import Sequence
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import ArrayLike, Bool, Float, Real, ScalarLike

from convax.sets._results import AxisAlignedBounds, SupportResult


class AbstractConvexSet(eqx.Module):
    """Base interface for convex set types.

    Attributes:
        ambient_dimension: Dimension of the containing vector space.
        dtype: JAX dtype used by the set.
    """

    ambient_dimension: eqx.AbstractVar[int]
    dtype: eqx.AbstractVar[DTypeLike]


class AbstractTranslationClosedSet(AbstractConvexSet):
    """Interface for set types closed under translation.

    Attributes:
        ambient_dimension: Dimension of the containing vector space.
        dtype: JAX dtype used by the set.
    """

    @abstractmethod
    def translate(
        self,
        offset: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> Self:
        """Return the translated set with the same concrete type.

        Args:
            offset: Translation vector with shape ``(ambient_dimension,)``.
        """


class AbstractNegationClosedSet(AbstractConvexSet):
    """Interface for set types closed under negation.

    Attributes:
        ambient_dimension: Dimension of the containing vector space.
        dtype: JAX dtype used by the set.
    """

    @abstractmethod
    def negate(self) -> Self:
        """Return the negated set with the same concrete type."""


class AbstractAffineMapClosedSet(
    AbstractTranslationClosedSet,
    AbstractNegationClosedSet,
):
    """Interface for set types closed under affine maps.

    Attributes:
        ambient_dimension: Dimension of the containing vector space.
        dtype: JAX dtype used by the set.
    """

    @abstractmethod
    def affine_map(
        self,
        matrix: Real[ArrayLike, "output_dimension {self.ambient_dimension}"]
        | Sequence[Sequence[float | int]],
        offset: Real[ArrayLike, "output_dimension"]
        | Sequence[float | int]
        | None = None,
    ) -> Self:
        """Return the affine image with the same concrete type.

        Args:
            matrix: Linear-map matrix with shape
                ``(output_dimension, ambient_dimension)``.
            offset: Optional translation vector with shape ``(output_dimension,)``;
                ``None`` selects zero.
        """

    def translate(
        self,
        offset: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> Self:
        """Return the translated set with the same concrete type.

        Args:
            offset: Translation vector with shape ``(ambient_dimension,)``.
        """
        return self.affine_map(
            jnp.eye(self.ambient_dimension, dtype=self.dtype),
            offset,
        )

    def negate(self) -> Self:
        return self.affine_map(-jnp.eye(self.ambient_dimension, dtype=self.dtype))


class AbstractAffinePreimageClosedSet(AbstractConvexSet):
    """Interface for set types closed under affine preimages.

    Attributes:
        ambient_dimension: Dimension of the containing vector space.
        dtype: JAX dtype used by the set.
    """

    @abstractmethod
    def affine_preimage(
        self,
        matrix: Real[ArrayLike, "{self.ambient_dimension} input_dimension"]
        | Sequence[Sequence[float | int]],
        offset: Real[ArrayLike, "{self.ambient_dimension}"]
        | Sequence[float | int]
        | None = None,
    ) -> Self:
        """Return the affine preimage with the same concrete type.

        Args:
            matrix: Linear-map matrix with shape
                ``(ambient_dimension, input_dimension)``.
            offset: Optional vector with shape ``(ambient_dimension,)`` added
                before membership testing; ``None`` selects zero.
        """


class AbstractMinkowskiSumClosedSet(AbstractConvexSet):
    """Interface for set types closed under Minkowski addition.

    Attributes:
        ambient_dimension: Dimension of the containing vector space.
        dtype: JAX dtype used by the set.
    """

    @abstractmethod
    def minkowski_sum(self, other: Self) -> Self:
        """Return the Minkowski sum with the same concrete type.

        Args:
            other: Operand with the same concrete type and ambient dimension.
        """


class AbstractIntersectionClosedSet(AbstractConvexSet):
    """Interface for set types closed under intersection.

    Attributes:
        ambient_dimension: Dimension of the containing vector space.
        dtype: JAX dtype used by the set.
    """

    @abstractmethod
    def intersection(self, other: Self) -> Self:
        """Return the intersection with the same concrete type.

        Args:
            other: Operand with the same concrete type and ambient dimension.
        """


class AbstractConvexHullClosedSet(AbstractConvexSet):
    """Interface for set types closed under convex hulls.

    Attributes:
        ambient_dimension: Dimension of the containing vector space.
        dtype: JAX dtype used by the set.
    """

    @abstractmethod
    def convex_hull(self, other: Self) -> Self:
        """Return the convex hull with the same concrete type.

        Args:
            other: Operand with the same concrete type and ambient dimension.
        """


class AbstractSupportSet(AbstractConvexSet):
    """Interface for compact convex sets with support functions.

    Attributes:
        ambient_dimension: Dimension of the containing vector space.
        dtype: JAX dtype used by the set.
    """

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
    """Interface for set types supporting point-containment queries.

    Attributes:
        ambient_dimension: Dimension of the containing vector space.
        dtype: JAX dtype used by the set.
    """

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
            EquinoxRuntimeError: If ``point`` or ``tolerance`` is nonfinite,
                or if ``tolerance`` is negative.
        """
