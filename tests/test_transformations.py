import jax
import jax.numpy as jnp

from convax import (
    AffineImage,
    Ellipsoid,
    HalfspacePolyhedron,
    convex_hull,
    intersection,
    minkowski_sum,
)


def test_composed_pipeline_compiles_as_one_region() -> None:
    left = Ellipsoid([0, 0], jnp.eye(2))
    right = Ellipsoid([1, 0], jnp.eye(2) * 0.5)
    matrix = jnp.array([[2.0, -1.0], [0.5, 1.0]])
    offset = jnp.array([1.0, -2.0])
    direction = jnp.array([1.0, 2.0])

    def pipeline(left_set: Ellipsoid, right_set: Ellipsoid):
        combined = convex_hull(left_set, minkowski_sum(left_set, right_set))
        return AffineImage(combined, matrix, offset).support(direction)

    eager_result = pipeline(left, right)
    compiled_result = jax.jit(pipeline)(left, right)

    assert jnp.allclose(compiled_result.value, eager_result.value)
    assert jnp.allclose(compiled_result.point, eager_result.point)


def test_vmap_batches_homogeneous_sets() -> None:
    centers = jnp.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 3.0]])
    generator_matrix = jnp.eye(2)
    direction = jnp.array([1.0, -2.0])

    batched_ellipsoids = jax.vmap(lambda center: Ellipsoid(center, generator_matrix))(
        centers
    )
    support_values = jax.jit(
        jax.vmap(lambda ellipsoid: ellipsoid.support_value(direction))
    )(batched_ellipsoids)

    expected = centers @ direction + jnp.linalg.norm(direction)
    assert jnp.allclose(support_values, expected)


def test_grad_flows_through_composite_parameters() -> None:
    direction = jnp.array([1.0, 2.0])

    def loss(offset: jax.Array) -> jax.Array:
        ellipsoid = Ellipsoid([0, 0], jnp.eye(2))
        return ellipsoid.affine_map(jnp.eye(2), offset).support_value(direction)

    gradient = jax.jit(jax.grad(loss))(jnp.zeros(2))

    assert jnp.allclose(gradient, direction)


def test_exact_halfspace_operations_compile() -> None:
    left = HalfspacePolyhedron([[1, 0]], [1])
    right = HalfspacePolyhedron([[-1, 0]], [1])
    matrix = jnp.array([[2.0, 0.0], [0.0, 0.5]])
    offset = jnp.array([0.5, -0.5])

    preimage = jax.jit(
        lambda convex_set, affine_matrix, affine_offset: convex_set.affine_preimage(
            affine_matrix, affine_offset
        )
    )(left, matrix, offset)
    combined = jax.jit(intersection)(left, right)
    translated = jax.jit(
        lambda convex_set, translation: convex_set.translate(translation)
    )(left, offset)
    reflected = jax.jit(lambda convex_set: convex_set.negate())(left)

    assert jnp.allclose(preimage.inequality_matrix, left.inequality_matrix @ matrix)
    assert combined.inequality_matrix.shape == (2, 2)
    assert translated.contains(offset)
    assert reflected.contains(jnp.array([-1.0, 0.0]))
