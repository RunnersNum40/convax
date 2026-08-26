import jax
import jax.numpy as jnp
import pytest
from jax.typing import DTypeLike
from jaxtyping import TypeCheckError

from convax.operations import affine_map
from convax.sets import Ellipsoid


def test_support_matches_closed_form() -> None:
    ellipsoid = Ellipsoid([1, -1], [[2, 0], [0, 1]])

    support = ellipsoid.support(jnp.array([1.0, 0.0]))

    assert jnp.allclose(support.value, 3.0)
    assert jnp.allclose(support.point, jnp.array([3.0, -1.0]))


def test_zero_direction_returns_center() -> None:
    ellipsoid = Ellipsoid([1, -1], [[2, 0], [0, 1]])

    support = ellipsoid.support(jnp.zeros(2))

    assert jnp.allclose(support.value, 0.0)
    assert jnp.allclose(support.point, ellipsoid.center)


@pytest.mark.parametrize("direction_magnitude", [1e-30, 1e20])
def test_support_norm_is_stable_at_float32_extremes(
    direction_magnitude: float,
) -> None:
    ellipsoid = Ellipsoid(
        jnp.zeros(1, dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
    )
    direction = jnp.array([direction_magnitude], dtype=jnp.float32)

    eager = ellipsoid.support(direction)
    compiled = jax.jit(
        lambda candidate_set, candidate_direction: candidate_set.support(
            candidate_direction
        )
    )(ellipsoid, direction)

    for support in (eager, compiled):
        assert jnp.isfinite(support.value)
        assert jnp.array_equal(support.value, direction[0])
        assert jnp.array_equal(support.point, jnp.ones(1, dtype=jnp.float32))


@pytest.mark.parametrize(
    ("dtype", "direction_magnitude"),
    [(jnp.float16, 300.0), (jnp.bfloat16, 1e20)],
    ids=["float16", "bfloat16"],
)
def test_support_norm_accumulates_low_precision_in_float32(
    dtype: DTypeLike,
    direction_magnitude: float,
) -> None:
    ellipsoid = Ellipsoid(
        jnp.zeros(1, dtype=dtype),
        jnp.ones((1, 1), dtype=dtype),
    )
    direction = jnp.array([direction_magnitude], dtype=dtype)

    eager = ellipsoid.support(direction)
    compiled = jax.jit(
        lambda candidate_set, candidate_direction: candidate_set.support(
            candidate_direction
        )
    )(ellipsoid, direction)

    for support in (eager, compiled):
        assert support.value.dtype == dtype
        assert support.point.dtype == dtype
        assert jnp.array_equal(support.value, direction[0])
        assert jnp.array_equal(support.point, jnp.ones(1, dtype=dtype))


def test_singleton_support_handles_empty_latent_dimension() -> None:
    singleton = Ellipsoid(jnp.array([2.0]), jnp.empty((1, 0)))
    direction = jnp.array([3.0])

    eager = singleton.support(direction)
    compiled = jax.jit(
        lambda candidate_set, candidate_direction: candidate_set.support(
            candidate_direction
        )
    )(singleton, direction)

    for support in (eager, compiled):
        assert jnp.array_equal(support.value, jnp.array(6.0))
        assert jnp.array_equal(support.point, singleton.center)


def test_rank_deficient_containment() -> None:
    ellipsoid = Ellipsoid([0, 0], [[1], [0]])

    assert ellipsoid.contains(jnp.array([0.5, 0.0]))
    assert not ellipsoid.contains(jnp.array([2.0, 0.0]))
    assert not ellipsoid.contains(jnp.array([0.5, 0.1]))


def test_singleton_containment() -> None:
    singleton = Ellipsoid(jnp.array([1.0, 2.0]), jnp.empty((2, 0)))

    assert singleton.contains(jnp.array([1.0, 2.0]))
    assert not singleton.contains(jnp.array([1.0, 2.1]))


def test_singleton_containment_detects_tiny_displacement() -> None:
    singleton = Ellipsoid(jnp.zeros(1), jnp.empty((1, 0)))
    point = jnp.array([1e-30], dtype=jnp.float32)
    tolerance = jnp.array(0.0, dtype=jnp.float32)

    assert not singleton.contains(point, tolerance=tolerance)
    assert not jax.jit(
        lambda candidate_point: singleton.contains(candidate_point, tolerance=tolerance)
    )(point)


@pytest.mark.parametrize(
    ("residual", "tolerance"),
    [(1e-30, 0.0), (1e20, 1e-3)],
)
def test_rank_deficient_containment_detects_extreme_off_hull_residuals(
    residual: float,
    tolerance: float,
) -> None:
    ellipsoid = Ellipsoid(
        jnp.zeros(2, dtype=jnp.float32),
        jnp.array([[1.0], [0.0]], dtype=jnp.float32),
    )
    point = jnp.array([0.0, residual], dtype=jnp.float32)

    assert not ellipsoid.contains(point, tolerance=tolerance)
    assert not jax.jit(
        lambda candidate_point: ellipsoid.contains(candidate_point, tolerance=tolerance)
    )(point)


def test_affine_map_preserves_ellipsoid_representation() -> None:
    ellipsoid = Ellipsoid([1, -1], [[2, 0], [0, 1]])
    matrix = jnp.array([[1.0, 2.0]])
    offset = jnp.array([0.5])

    eager = affine_map(ellipsoid, matrix, offset)
    compiled = jax.jit(affine_map)(ellipsoid, matrix, offset)

    assert isinstance(eager, Ellipsoid)
    assert jnp.array_equal(eager.center, matrix @ ellipsoid.center + offset)
    assert jnp.array_equal(eager.generator_matrix, matrix @ ellipsoid.generator_matrix)
    assert jnp.array_equal(compiled.center, eager.center)
    assert jnp.array_equal(compiled.generator_matrix, eager.generator_matrix)


def test_support_is_jittable_vectorizable_and_differentiable() -> None:
    center = jnp.array([1.0, -1.0])
    generator_matrix = jnp.array([[2.0, 0.0], [0.0, 1.0]])
    ellipsoid = Ellipsoid(center, generator_matrix)
    directions = jnp.eye(2)

    eager_values = jax.vmap(ellipsoid.support_value)(directions)
    compiled_values = jax.jit(jax.vmap(ellipsoid.support_value))(directions)
    center_gradient = jax.grad(
        lambda candidate_center: Ellipsoid(
            candidate_center, generator_matrix
        ).support_value(jnp.array([1.0, 2.0]))
    )(center)

    assert jnp.allclose(compiled_values, eager_values)
    assert jnp.allclose(center_gradient, jnp.array([1.0, 2.0]))


def test_support_direction_gradient_uses_stable_norm() -> None:
    ellipsoid = Ellipsoid(jnp.zeros(2), jnp.eye(2))
    direction = jnp.array([3.0, 4.0])

    direction_gradient = jax.grad(ellipsoid.support_value)(direction)

    assert jnp.allclose(direction_gradient, jnp.array([0.6, 0.8]))


@pytest.mark.parametrize(
    "dtype", [jnp.float16, jnp.bfloat16], ids=["float16", "bfloat16"]
)
def test_support_direction_gradient_with_low_precision(dtype: DTypeLike) -> None:
    ellipsoid = Ellipsoid(jnp.zeros(2, dtype=dtype), jnp.eye(2, dtype=dtype))
    direction = jnp.array([3.0, 4.0], dtype=dtype)

    direction_gradient = jax.grad(ellipsoid.support_value)(direction)

    assert direction_gradient.dtype == dtype
    assert jnp.allclose(
        direction_gradient.astype(jnp.float32),
        jnp.array([0.6, 0.8]),
        rtol=1e-2,
        atol=1e-2,
    )


def test_constructor_rejects_incompatible_shapes() -> None:
    with pytest.raises(TypeCheckError, match="parameter 'generator_matrix'"):
        Ellipsoid(jnp.zeros(2), jnp.zeros((3, 1)))

    with pytest.raises(TypeCheckError, match="parameter 'center'"):
        Ellipsoid(jnp.zeros((1, 2)), jnp.zeros((2, 1)))
