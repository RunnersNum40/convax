import jax.numpy as jnp

from convax.sets import (
    AbstractSupportSet,
    ConvexHull,
    HalfspacePolyhedron,
    MinkowskiSum,
)


def convex_hull(
    left_set: AbstractSupportSet, right_set: AbstractSupportSet
) -> ConvexHull:
    """Return the convex hull of two compact convex sets."""
    return ConvexHull(left_set, right_set)


def intersection(
    left_set: HalfspacePolyhedron, right_set: HalfspacePolyhedron
) -> HalfspacePolyhedron:
    """Intersect two halfspace polyhedra without removing redundancies."""
    if left_set.ambient_dimension != right_set.ambient_dimension:
        raise ValueError(
            "intersection dimensions must match, got "
            f"{left_set.ambient_dimension} and {right_set.ambient_dimension}"
        )
    return HalfspacePolyhedron(
        jnp.concatenate(
            (left_set.inequality_matrix, right_set.inequality_matrix), axis=0
        ),
        jnp.concatenate(
            (left_set.inequality_bounds, right_set.inequality_bounds), axis=0
        ),
        jnp.concatenate((left_set.equality_matrix, right_set.equality_matrix), axis=0),
        jnp.concatenate((left_set.equality_values, right_set.equality_values), axis=0),
    )


def minkowski_sum(
    left_set: AbstractSupportSet, right_set: AbstractSupportSet
) -> MinkowskiSum:
    """Return the Minkowski sum of two compact convex sets."""
    return MinkowskiSum(left_set, right_set)
