from __future__ import annotations

from abc import abstractmethod
from typing import Any, Self

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, ScalarLike


class AbstractSet(eqx.Module):
    """
    Base class for convex sets.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Dimension of the ambient space.

        Returns:
            Dimension of the ambient space.
        """

    @property
    @abstractmethod
    def is_empty(self) -> Bool[Array, ""]:
        """
        Whether the set is empty.

        Returns:
            True if the set is empty, otherwise False.
        """

    @property
    @abstractmethod
    def is_bounded(self) -> Bool[Array, ""]:
        """
        Whether the set is bounded.

        Returns:
            True if the set is bounded, otherwise False.
        """

    @abstractmethod
    def contains(self, x: Any) -> Bool[Array, ""]:
        """
        Return whether the set contains the point `x`.

        Args:
            x: Point to test for membership.

        Returns:
            True if `x` is in the set, otherwise False.
        """

    def __contains__(self, x: Any) -> Bool[Array, ""]:
        """
        Alias for `contains`.

        Args:
            x: Point to test for membership.

        Returns:
            True if `x` is in the set, otherwise False.
        """
        return self.contains(x)

    @abstractmethod
    def support_point(self, direction: Float[ArrayLike, " n"]) -> Float[Array, " n"]:
        """
        Return a maximizer of `direction @ y` over `y` in the set.

        Args:
            direction: Direction vector.

        Returns:
            A point `y` in the set that maximizes `direction @ y`.

        Raises:
            ValueError: If the direction does not have shape `(n,)` where `n == dim`.
            ValueError: If the support problem is not well-defined for this set (e.g.
                the set is unbounded in `direction`).
        """

    def support_function(self, direction: Float[ArrayLike, " n"]) -> Float[Array, ""]:
        """
        Return `max_{y in S} direction @ y`.

        Default implementation uses `support_point`.

        Args:
            direction: Direction vector.

        Returns:
            The maximum value of `direction @ y` over `y` in the set.

        Raises:
            ValueError: If the direction does not have shape `(n,)` where `n == dim`.
            ValueError: If the support problem is not well-defined for this set (e.g.
                the set is unbounded in `direction`).
        """
        direction_arr = jnp.asarray(direction)
        argmax_point = self.support_point(direction_arr)
        return jnp.asarray(direction_arr @ argmax_point)

    @abstractmethod
    def closest_point(
        self, x: Float[ArrayLike, " n"], p: ScalarLike = 2
    ) -> Float[Array, " n"]:
        """
        Return a point in the set closest to `x` in the `p`-norm.

        Args:
            x: Query point.
            p: Norm parameter.

        Returns:
            A point in the set minimizing `||y - x||_p`.

        Raises:
            ValueError: If `x` does not have shape `(n,)` where `n == dim`.
            ValueError: If `p` is not a supported norm.
            ValueError: If the set is empty.
        """

    def distance(
        self, x: Float[ArrayLike, " n"], p: ScalarLike = 2
    ) -> Float[Array, ""]:
        """
        Return the distance from `x` to the set in the `p`-norm.

        Default implementation uses `closest_point`.

        Args:
            x: Query point.
            p: Norm parameter.

        Returns:
            The distance `min_{y in S} ||y - x||_p`.

        Raises:
            ValueError: If `x` does not have shape `(n,)` where `n == dim`.
            ValueError: If `p` is not a supported norm.
            ValueError: If the set is empty.
        """
        x_arr = jnp.asarray(x)
        projection = self.closest_point(x_arr, p=p)
        return jnp.linalg.norm(x_arr - projection, ord=p)

    def indicator(self, x: Float[ArrayLike, " n"]) -> Float[Array, ""]:
        """
        Indicator function of the set.

        Returns `0` if `x` is in the set and `+inf` otherwise.

        Args:
            x: Query point.

        Returns:
            `0` if `x` is in the set and `+inf` otherwise.

        Raises:
            ValueError: If `x` does not have shape `(n,)` where `n == dim`.
        """
        return jnp.where(self.contains(x), 0.0, jnp.inf)

    @abstractmethod
    def linear_map(self, M: Float[ArrayLike, " m n"] | Float[ArrayLike, ""]) -> Self:
        """
        Return the image of the set under a linear map.

        Computes `{ M @ y : y in S }` (or scalar scaling if `M` is a scalar).

        Args:
            M: Linear map as a matrix, or a scalar for uniform scaling.

        Returns:
            The linearly transformed set.

        Raises:
            ValueError: If `M` has incompatible shape.
        """

    @abstractmethod
    def translate(self, b: Float[ArrayLike, " n"]) -> Self:
        """
        Return the translation of the set by `b`.

        Computes `{ y + b : y in S }`.

        Args:
            b: Translation vector.

        Returns:
            The translated set.

        Raises:
            ValueError: If `b` does not have shape `(n,)` where `n == dim`.
        """

    def affine_map(
        self,
        M: Float[ArrayLike, " m n"] | Float[ArrayLike, ""],
        b: Float[ArrayLike, " m"] | None = None,
    ) -> Self:
        """
        Return the image of the set under an affine map.

        Computes `{ M @ y + b : y in S }`.

        Default implementation composes `linear_map` and `translate`.

        Args:
            M: Linear map as a matrix, or a scalar for uniform scaling.
            b: Translation vector. If None, returns only the linear image.

        Returns:
            The affinely transformed set.

        Raises:
            ValueError: If `M` has incompatible shape.
            ValueError: If `b` has incompatible shape.
        """
        mapped = self.linear_map(M)
        if b is None:
            return mapped
        return mapped.translate(b)

    def __rmul__(self, scalar: Float[ArrayLike, ""]) -> Self:
        """
        Right scalar multiplication (uniform scaling).

        Args:
            scalar: Scalar multiplier.

        Returns:
            The scaled set.
        """
        return self.linear_map(scalar)

    def __rmatmul__(self, matrix: Float[ArrayLike, " m n"]) -> Self:
        """
        Right matrix multiplication (linear map).

        Args:
            matrix: Matrix multiplier.

        Returns:
            The linearly transformed set.
        """
        return self.linear_map(matrix)

    def __add__(self, shift: Float[ArrayLike, " n"]) -> Self:
        """
        Translate by `shift`.

        Args:
            shift: Translation vector.

        Returns:
            The translated set.
        """
        return self.translate(shift)

    def __sub__(self, shift: Float[ArrayLike, " n"]) -> Self:
        """
        Translate by `-shift`.

        Args:
            shift: Translation vector.

        Returns:
            The translated set.
        """
        return self.translate(-jnp.asarray(shift))

    def __neg__(self) -> Self:
        """
        Return the negation of the set.

        Returns:
            The negated set.
        """
        return self.linear_map(-1)
