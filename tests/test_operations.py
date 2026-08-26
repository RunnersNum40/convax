from typing import Any, assert_type, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from convax._utils import IntegerVectorLike
from convax.operations import (
    affine_map,
    affine_preimage,
    convex_hull,
    intersection,
    minkowski_sum,
    negate,
    project_coordinates,
    translate,
)
from convax.sets import (
    AbstractAffineMapClosedSet,
    AbstractNegationClosedSet,
    AbstractSupportSet,
    AbstractTranslationClosedSet,
    AffineImage,
    ConstrainedZonotope,
    ConvexHull,
    Ellipsoid,
    HalfspacePolyhedron,
    MinkowskiSum,
    VertexPolytope,
    Zonotope,
)


def translate_capability[SetT: AbstractTranslationClosedSet](
    convex_set: SetT,
    offset: jax.Array,
) -> SetT:
    return translate(convex_set, offset)


def negate_capability[SetT: AbstractNegationClosedSet](convex_set: SetT) -> SetT:
    return negate(convex_set)


def test_affine_map_transforms_support_value_and_point() -> None:
    ellipsoid = Ellipsoid([0, 0], jnp.eye(2))
    image = affine_map(ellipsoid, [[2, 0], [0, 3]], [1, -1])

    support = image.support(jnp.array([1.0, 0.0]))

    assert isinstance(image, Ellipsoid)
    assert jnp.array_equal(image.center, jnp.array([1.0, -1.0]))
    assert jnp.array_equal(image.generator_matrix, jnp.diag(jnp.array([2.0, 3.0])))
    assert jnp.allclose(support.value, 3.0)
    assert jnp.allclose(support.point, jnp.array([3.0, -1.0]))


def test_affine_preimage_matches_definition() -> None:
    polyhedron = HalfspacePolyhedron(jnp.eye(2), jnp.ones(2))
    matrix = jnp.array([[2.0, 0.0], [0.0, 0.5]])
    offset = jnp.array([0.5, -0.5])
    preimage = affine_preimage(polyhedron, matrix, offset)
    point = jnp.array([0.2, 1.0])

    assert preimage.contains(point) == polyhedron.contains(matrix @ point + offset)


def test_affine_preimage_promotes_before_arithmetic() -> None:
    polyhedron = HalfspacePolyhedron(
        jnp.ones((1, 1), dtype=jnp.float16),
        jnp.ones(1, dtype=jnp.float16),
    )
    matrix = jnp.ones((1, 1), dtype=jnp.float32)
    offset = jnp.array([0.0004], dtype=jnp.float16)
    expected_bound = polyhedron.inequality_bounds.astype(jnp.float32) - (
        polyhedron.inequality_matrix.astype(jnp.float32) @ offset.astype(jnp.float32)
    )

    eager = affine_preimage(polyhedron, matrix, offset)
    compiled = jax.jit(affine_preimage)(polyhedron, matrix, offset)

    assert eager.inequality_bounds.dtype == jnp.float32
    assert jnp.array_equal(eager.inequality_bounds, expected_bound)
    assert jnp.array_equal(compiled.inequality_bounds, expected_bound)


def test_affine_preimage_defaults_zero_offset_and_rejects_dimension_mismatch() -> None:
    polyhedron = HalfspacePolyhedron([[1, 0]], [1])

    preimage = affine_preimage(polyhedron, [[2], [0]])

    assert jnp.array_equal(preimage.inequality_matrix, jnp.array([[2.0]]))
    assert jnp.array_equal(preimage.inequality_bounds, polyhedron.inequality_bounds)
    with pytest.raises(TypeCheckError, match="parameter 'matrix'"):
        affine_preimage(polyhedron, jnp.eye(3))


def test_coordinate_projection_preserves_order_and_duplicates() -> None:
    zonotope = Zonotope([1, 2, 3], jnp.eye(3))

    projected = project_coordinates(zonotope, jnp.array([2, 0, 2]))

    assert isinstance(projected, Zonotope)
    assert projected.ambient_dimension == 3
    assert jnp.array_equal(projected.center, jnp.array([3.0, 1.0, 3.0]))
    assert jnp.array_equal(
        projected.generator_matrix,
        jnp.array(
            [[0, 0, 1], [1, 0, 0], [0, 0, 1]],
            dtype=zonotope.dtype,
        ),
    )


def test_coordinate_projection_accepts_empty_integer_coordinates() -> None:
    zonotope = Zonotope([1, 2, 3], jnp.eye(3))

    projected = project_coordinates(zonotope, jnp.array([], dtype=jnp.int32))

    assert projected.center.shape == (0,)
    assert projected.generator_matrix.shape == (0, 3)


def test_coordinate_projection_matrix_matches_float16_set_dtype() -> None:
    zonotope = Zonotope(
        jnp.array([1, 2, 3], dtype=jnp.float16),
        jnp.eye(3, dtype=jnp.float16),
    )

    projected = project_coordinates(zonotope, jnp.array([2, 0]))

    assert projected.generator_matrix.dtype == zonotope.dtype


def test_coordinate_projection_rejects_invalid_rank_and_dtype() -> None:
    zonotope = Zonotope([1, 2], jnp.eye(2))

    with pytest.raises(TypeCheckError, match="parameter 'coordinates'"):
        project_coordinates(zonotope, cast(IntegerVectorLike, [[0]]))
    with pytest.raises(TypeCheckError, match="parameter 'coordinates'"):
        project_coordinates(zonotope, cast(IntegerVectorLike, [0.0]))


@pytest.mark.parametrize("invalid_coordinate", [-1, 2])
def test_coordinate_projection_rejects_invalid_indices_eagerly(
    invalid_coordinate: int,
) -> None:
    zonotope = Zonotope([1, 2], jnp.eye(2))

    with pytest.raises(eqx.EquinoxRuntimeError, match="ambient dimension"):
        project_coordinates(zonotope, jnp.array([invalid_coordinate]))


def test_coordinate_projection_rejects_invalid_indices_under_jit() -> None:
    zonotope = Zonotope([1, 2], jnp.eye(2))
    compiled_projection = jax.jit(project_coordinates)

    with pytest.raises(jax.errors.JaxRuntimeError, match="ambient dimension"):
        compiled_projection(zonotope, jnp.array([2]))


def test_minkowski_sum_adds_support_results() -> None:
    ellipsoid = Ellipsoid([0, 0], jnp.eye(2))
    zonotope = Zonotope([1, 0], jnp.eye(2))
    direction = jnp.array([1.0, 2.0])

    summed = minkowski_sum(ellipsoid, zonotope)
    summed_support = summed.support(direction)
    left_support = ellipsoid.support(direction)
    right_support = zonotope.support(direction)

    assert_type(summed, AbstractSupportSet)
    assert isinstance(summed, MinkowskiSum)
    assert jnp.allclose(summed_support.value, left_support.value + right_support.value)
    assert jnp.allclose(summed_support.point, left_support.point + right_support.point)


def test_convex_hull_selects_larger_support() -> None:
    left = VertexPolytope([[0, 0], [1, 0]])
    right = VertexPolytope([[0, 0], [3, 0]])

    hull = convex_hull(left, right)
    support = jax.jit(lambda hull: hull.support(jnp.array([1.0, 0.0])))(hull)

    assert_type(hull, VertexPolytope)
    assert isinstance(hull, VertexPolytope)
    assert jnp.array_equal(
        hull.vertices, jnp.concatenate((left.vertices, right.vertices))
    )
    assert jnp.allclose(support.value, 3.0)
    assert jnp.allclose(support.point, jnp.array([3.0, 0.0]))


def test_halfspace_intersection_concatenates_constraints() -> None:
    left = HalfspacePolyhedron([[1, 0]], [1], [[0, 1]], [0])
    right = HalfspacePolyhedron([[-1, 0]], [1], [[1, 1]], [1])

    result = intersection(left, right)

    assert result.inequality_matrix.shape == (2, 2)
    assert result.equality_matrix.shape == (2, 2)
    assert result.contains(jnp.array([1.0, 0.0]))


def test_closed_operation_methods_preserve_concrete_representations() -> None:
    ellipsoid = Ellipsoid([0], [[1]])
    halfspace_polyhedron = HalfspacePolyhedron([[1]], [1])
    constrained_zonotope = ConstrainedZonotope(
        [0], [[1]], jnp.empty((0, 1)), jnp.empty((0,))
    )
    vertex_polytope = VertexPolytope([[0], [1]])

    assert isinstance(ellipsoid.affine_map([[2]]), Ellipsoid)
    assert isinstance(ellipsoid.translate([1]), Ellipsoid)
    assert isinstance(ellipsoid.negate(), Ellipsoid)
    assert isinstance(halfspace_polyhedron.affine_preimage([[2]]), HalfspacePolyhedron)
    assert isinstance(
        halfspace_polyhedron.intersection(halfspace_polyhedron),
        HalfspacePolyhedron,
    )
    assert isinstance(
        constrained_zonotope.minkowski_sum(constrained_zonotope),
        ConstrainedZonotope,
    )
    assert isinstance(vertex_polytope.convex_hull(vertex_polytope), VertexPolytope)


def test_translation_supports_compact_and_halfspace_sets() -> None:
    zonotope = Zonotope([0, 0], jnp.eye(2))
    polyhedron = HalfspacePolyhedron(jnp.eye(2), jnp.ones(2))
    offset = jnp.array([2.0, -1.0])

    translated_zonotope = translate_capability(zonotope, offset)
    translated_polyhedron = translate_capability(polyhedron, offset)

    assert isinstance(translated_zonotope, Zonotope)
    assert jnp.array_equal(translated_zonotope.center, offset)
    assert jnp.allclose(
        translated_zonotope.support_point(jnp.array([1.0, 0.0])),
        jnp.array([3.0, -1.0]),
    )
    assert translated_polyhedron.contains(offset)


def test_negation_reflects_compact_and_halfspace_sets() -> None:
    zonotope = Zonotope([1, 0], jnp.eye(2))
    polyhedron = HalfspacePolyhedron([[1, 0]], [2])

    reflected_zonotope = negate_capability(zonotope)
    reflected_polyhedron = negate_capability(polyhedron)

    assert isinstance(reflected_zonotope, Zonotope)
    assert jnp.array_equal(reflected_zonotope.center, jnp.array([-1.0, 0.0]))
    assert jnp.allclose(
        reflected_zonotope.support_point(jnp.array([1.0, 0.0])), jnp.zeros(2)
    )
    assert reflected_polyhedron.contains(jnp.array([-2.0, 0.0]))
    assert not reflected_polyhedron.contains(jnp.array([-3.0, 0.0]))


def test_composite_transformations_use_affine_image_fallbacks() -> None:
    left = Ellipsoid([0, 0], jnp.eye(2))
    right = Ellipsoid([1, 0], jnp.eye(2))
    hull = convex_hull(left, right)

    assert_type(hull, AbstractSupportSet)
    assert isinstance(hull, ConvexHull)
    assert not isinstance(hull, AbstractAffineMapClosedSet)
    assert not isinstance(hull, AbstractTranslationClosedSet)
    assert not isinstance(hull, AbstractNegationClosedSet)
    assert isinstance(affine_map(hull, [[1, 0]]), AffineImage)
    assert isinstance(project_coordinates(hull, [0]), AffineImage)
    assert isinstance(translate(hull, [1, 0]), AffineImage)
    assert isinstance(negate(hull), AffineImage)


def test_operation_dimension_mismatches_fail_loudly() -> None:
    one_dimensional = Zonotope([0], jnp.ones((1, 1)))
    two_dimensional = Zonotope([0, 0], jnp.eye(2))
    one_dimensional_polytope = VertexPolytope([[0]])
    two_dimensional_polytope = VertexPolytope([[0, 0]])

    with pytest.raises(ValueError, match="dimensions must match"):
        minkowski_sum(one_dimensional, two_dimensional)

    with pytest.raises(ValueError, match="dimensions must match"):
        convex_hull(one_dimensional, two_dimensional)

    with pytest.raises(ValueError, match="dimensions must match"):
        convex_hull(one_dimensional_polytope, two_dimensional_polytope)


def test_unsupported_operation_combinations_fail_loudly() -> None:
    constrained_zonotope = ConstrainedZonotope(
        [0], [[1]], jnp.empty((0, 1)), jnp.empty((0,))
    )
    halfspace_polyhedron = HalfspacePolyhedron([[1]], [1])

    with pytest.raises(TypeError, match="convex hull is not implemented"):
        cast(Any, convex_hull)(halfspace_polyhedron, halfspace_polyhedron)

    with pytest.raises(TypeError, match="Minkowski sum is not implemented"):
        cast(Any, minkowski_sum)(constrained_zonotope, halfspace_polyhedron)

    with pytest.raises(TypeError, match="intersection is not implemented"):
        cast(Any, intersection)(constrained_zonotope, halfspace_polyhedron)
    with pytest.raises(TypeError, match="affine map is not implemented"):
        cast(Any, affine_map)(halfspace_polyhedron, [[1]])
    with pytest.raises(TypeError, match="coordinate projection is not implemented"):
        cast(Any, project_coordinates)(halfspace_polyhedron, [0])
    with pytest.raises(TypeCheckError, match="parameter 'convex_set'"):
        cast(Any, affine_preimage)(Ellipsoid([0], [[1]]), [[1]])
    with pytest.raises(TypeCheckError, match="parameter 'left_set'"):
        cast(Any, intersection)(Ellipsoid([0], [[1]]), Ellipsoid([0], [[1]]))


def test_composites_preserve_promoted_query_precision() -> None:
    lower_precision = VertexPolytope(jnp.array([[0], [1]], dtype=jnp.float16))
    higher_precision = VertexPolytope(jnp.array([[0], [0.5]], dtype=jnp.float32))
    direction = jnp.array([1e-8], dtype=jnp.float32)

    image_support = affine_map(lower_precision, jnp.eye(1, dtype=jnp.float32)).support(
        direction
    )
    hull_support = convex_hull(lower_precision, higher_precision).support(direction)

    assert image_support.value.dtype == jnp.float32
    assert jnp.allclose(image_support.value, 1e-8)
    assert jnp.allclose(hull_support.value, 1e-8)
    assert jnp.allclose(hull_support.point, jnp.ones(1))


def test_affine_image_composes_maps_without_nesting() -> None:
    left = Ellipsoid([0, 0], jnp.eye(2))
    right = Ellipsoid([1, 0], jnp.eye(2))
    source = convex_hull(left, right)
    image = affine_map(source, [[2, 0], [0, 1]], [1, 0])

    mapped = affine_map(image, [[1, 1]], [3])

    assert isinstance(image, AffineImage)
    assert isinstance(mapped, AffineImage)
    assert mapped.convex_set is source
    assert jnp.array_equal(mapped.matrix, jnp.array([[2.0, 1.0]]))
    assert jnp.array_equal(mapped.offset, jnp.array([4.0]))
