from typing import overload

from convax.sets import (
    AbstractConvexHullClosedSet,
    AbstractConvexSet,
    AbstractIntersectionClosedSet,
    AbstractMinkowskiSumClosedSet,
    AbstractSupportSet,
    ConvexHull,
    MinkowskiSum,
)


@overload
def convex_hull[SetT: AbstractConvexHullClosedSet](
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

    Retain matching convex-hull-closed set types; otherwise return lazy
    ``ConvexHull`` for support-capable operands.

    Args:
        left_set: First convex-set operand.
        right_set: Second convex-set operand with the same ambient dimension.

    Raises:
        TypeError: If neither a type-preserving construction nor the
            support-function fallback is available.
        ValueError: If an available construction receives sets with different
            ambient dimensions.
    """
    if (
        type(left_set) is type(right_set)
        and isinstance(left_set, AbstractConvexHullClosedSet)
        and isinstance(right_set, AbstractConvexHullClosedSet)
    ):
        return left_set.convex_hull(right_set)
    if isinstance(left_set, AbstractSupportSet) and isinstance(
        right_set, AbstractSupportSet
    ):
        return ConvexHull(left_set, right_set)
    raise TypeError(
        "convex hull is not implemented for "
        f"{type(left_set).__name__} and {type(right_set).__name__}"
    )


def intersection[SetT: AbstractIntersectionClosedSet](
    left_set: SetT, right_set: SetT
) -> SetT:
    """Return the intersection of two sets.

    The set type must be intersection-closed because support functions generally
    do not determine intersections.

    Args:
        left_set: First intersection-closed convex-set operand.
        right_set: Second operand with the same concrete type and ambient
            dimension.

    Raises:
        TypeError: If the sets have different concrete types.
        ValueError: If matching set types have different ambient dimensions.
    """
    if (
        type(left_set) is not type(right_set)
        or not isinstance(left_set, AbstractIntersectionClosedSet)
        or not isinstance(right_set, AbstractIntersectionClosedSet)
    ):
        raise TypeError(
            "intersection is not implemented for "
            f"{type(left_set).__name__} and {type(right_set).__name__}"
        )
    return left_set.intersection(right_set)


@overload
def minkowski_sum[SetT: AbstractMinkowskiSumClosedSet](
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

    Retain matching Minkowski-addition-closed set types; otherwise return lazy
    ``MinkowskiSum`` for support-capable operands.

    Args:
        left_set: First convex-set operand.
        right_set: Second convex-set operand with the same ambient dimension.

    Raises:
        TypeError: If neither a type-preserving construction nor the
            support-function fallback is available.
        ValueError: If an available construction receives sets with different
            ambient dimensions.
    """
    if (
        type(left_set) is type(right_set)
        and isinstance(left_set, AbstractMinkowskiSumClosedSet)
        and isinstance(right_set, AbstractMinkowskiSumClosedSet)
    ):
        return left_set.minkowski_sum(right_set)
    if isinstance(left_set, AbstractSupportSet) and isinstance(
        right_set, AbstractSupportSet
    ):
        return MinkowskiSum(left_set, right_set)
    raise TypeError(
        "Minkowski sum is not implemented for "
        f"{type(left_set).__name__} and {type(right_set).__name__}"
    )
