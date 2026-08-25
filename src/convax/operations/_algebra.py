from typing import overload

from convax.sets import (
    AbstractConvexHullSet,
    AbstractConvexSet,
    AbstractIntersectionSet,
    AbstractMinkowskiSumSet,
    AbstractSupportSet,
    ConvexHull,
    MinkowskiSum,
)


@overload
def convex_hull[SetT: AbstractConvexHullSet](
    left_set: SetT, right_set: SetT
) -> SetT: ...


@overload
def convex_hull(
    left_set: AbstractSupportSet, right_set: AbstractSupportSet
) -> AbstractSupportSet: ...


def convex_hull(
    left_set: AbstractConvexSet, right_set: AbstractConvexSet
) -> AbstractConvexSet:
    """Return the smallest convex set containing both operands.

    Retain matching convex-hull-closed representations; otherwise, return a lazy ``ConvexHull``
    for support-capable operands.

    Args:
        left_set: First convex-set operand.
        right_set: Second convex-set operand with the same ambient dimension.

    Raises:
        TypeError: If neither representation-preserving construction nor the
            support-function fallback is available.
        ValueError: If an available construction receives sets with different
            ambient dimensions.
    """
    if (
        type(left_set) is type(right_set)
        and isinstance(left_set, AbstractConvexHullSet)
        and isinstance(right_set, AbstractConvexHullSet)
    ):
        return left_set._convex_hull(right_set)
    if isinstance(left_set, AbstractSupportSet) and isinstance(
        right_set, AbstractSupportSet
    ):
        return ConvexHull(left_set, right_set)
    raise TypeError(
        "convex hull is not implemented for "
        f"{type(left_set).__name__} and {type(right_set).__name__}"
    )


def intersection[SetT: AbstractIntersectionSet](
    left_set: SetT, right_set: SetT
) -> SetT:
    """Return the intersection of two sets.

    The representation must be intersection-closed because support functions generally do not
    determine intersections.

    Args:
        left_set: First intersection-closed convex-set operand.
        right_set: Second operand with the same concrete type and ambient
            dimension.

    Raises:
        TypeError: If the sets have different concrete types.
        ValueError: If matching representations have different ambient dimensions.
    """
    if (
        type(left_set) is not type(right_set)
        or not isinstance(left_set, AbstractIntersectionSet)
        or not isinstance(right_set, AbstractIntersectionSet)
    ):
        raise TypeError(
            "intersection is not implemented for "
            f"{type(left_set).__name__} and {type(right_set).__name__}"
        )
    return left_set._intersection(right_set)


@overload
def minkowski_sum[SetT: AbstractMinkowskiSumSet](
    left_set: SetT, right_set: SetT
) -> SetT: ...


@overload
def minkowski_sum(
    left_set: AbstractSupportSet, right_set: AbstractSupportSet
) -> AbstractSupportSet: ...


def minkowski_sum(
    left_set: AbstractConvexSet, right_set: AbstractConvexSet
) -> AbstractConvexSet:
    """Return the Minkowski sum, ``{x + y | x in left_set, y in right_set}``.

    Retain matching Minkowski-addition-closed representations; otherwise, return a lazy
    ``MinkowskiSum`` for support-capable operands.

    Args:
        left_set: First convex-set operand.
        right_set: Second convex-set operand with the same ambient dimension.

    Raises:
        TypeError: If neither representation-preserving construction nor the
            support-function fallback is available.
        ValueError: If an available construction receives sets with different
            ambient dimensions.
    """
    if (
        type(left_set) is type(right_set)
        and isinstance(left_set, AbstractMinkowskiSumSet)
        and isinstance(right_set, AbstractMinkowskiSumSet)
    ):
        return left_set._minkowski_sum(right_set)
    if isinstance(left_set, AbstractSupportSet) and isinstance(
        right_set, AbstractSupportSet
    ):
        return MinkowskiSum(left_set, right_set)
    raise TypeError(
        "Minkowski sum is not implemented for "
        f"{type(left_set).__name__} and {type(right_set).__name__}"
    )
