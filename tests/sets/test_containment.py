import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from convax import (
    AbstractPointContainmentSet,
    Ellipsoid,
    HalfspacePolyhedron,
)


@pytest.fixture(params=["ellipsoid", "polyhedron"])
def containment_set(request: pytest.FixtureRequest) -> AbstractPointContainmentSet:
    if request.param == "ellipsoid":
        return Ellipsoid([0], [[1]])
    return HalfspacePolyhedron([[1], [-1]], [1, 1])


def test_containment_accepts_explicit_scalar_tolerance(
    containment_set: AbstractPointContainmentSet,
) -> None:
    assert containment_set.contains([1.0001], tolerance=1e-3)


def test_containment_rejects_vector_tolerance(
    containment_set: AbstractPointContainmentSet,
) -> None:
    with pytest.raises(ValueError, match="tolerance must be a scalar"):
        containment_set.contains([0], tolerance=jnp.array([1e-3]))


def test_containment_rejects_complex_tolerance(
    containment_set: AbstractPointContainmentSet,
) -> None:
    with pytest.raises(TypeError, match="requires real-valued arrays"):
        containment_set.contains([0], tolerance=jnp.array(1e-3 + 1e-3j))


@pytest.mark.parametrize("tolerance", [-1e-3, jnp.inf, jnp.nan])
def test_containment_rejects_invalid_tolerance_eagerly(
    containment_set: AbstractPointContainmentSet,
    tolerance: float | jax.Array,
) -> None:
    with pytest.raises(eqx.EquinoxRuntimeError, match="finite and nonnegative"):
        containment_set.contains([0], tolerance=tolerance)


def test_containment_rejects_invalid_tolerance_under_jit(
    containment_set: AbstractPointContainmentSet,
) -> None:
    compiled_contains = jax.jit(
        lambda point, tolerance: containment_set.contains(point, tolerance=tolerance)
    )

    with pytest.raises(jax.errors.JaxRuntimeError, match="finite and nonnegative"):
        compiled_contains(jnp.zeros(1), jnp.array(-1e-3))


@pytest.mark.parametrize(
    "containment_set",
    [
        Ellipsoid(
            jnp.zeros(1, dtype=jnp.float16),
            jnp.empty((1, 0), dtype=jnp.float16),
        ),
        HalfspacePolyhedron(
            jnp.array([[1], [-1]], dtype=jnp.float16),
            jnp.ones(2, dtype=jnp.float16),
        ),
    ],
)
def test_containment_validates_tolerance_before_dtype_promotion(
    containment_set: AbstractPointContainmentSet,
) -> None:
    tolerance = jnp.array(-1e-8, dtype=jnp.float32)

    with pytest.raises(eqx.EquinoxRuntimeError, match="finite and nonnegative"):
        containment_set.contains([0], tolerance=tolerance)

    compiled_contains = jax.jit(
        lambda candidate_tolerance: containment_set.contains(
            [0], tolerance=candidate_tolerance
        )
    )
    with pytest.raises(jax.errors.JaxRuntimeError, match="finite and nonnegative"):
        compiled_contains(tolerance)


def test_halfspace_containment_promotes_all_operands_before_arithmetic() -> None:
    polyhedron = HalfspacePolyhedron(
        jnp.ones((1, 1), dtype=jnp.float16),
        jnp.ones(1, dtype=jnp.float16),
    )
    point = jnp.array([1.0003], dtype=jnp.float32)
    tolerance = jnp.array(0.0004, dtype=jnp.float16)

    assert polyhedron.contains(point, tolerance=tolerance)
    assert jax.jit(
        lambda candidate_set, candidate_point, candidate_tolerance: (
            candidate_set.contains(candidate_point, tolerance=candidate_tolerance)
        )
    )(polyhedron, point, tolerance)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="requires x64")
def test_ellipsoid_containment_promotes_generator_before_pseudoinverse() -> None:
    ellipsoid = Ellipsoid(
        jnp.zeros(2, dtype=jnp.float32),
        jnp.diag(jnp.array([1.0, 1e-8], dtype=jnp.float32)),
    )
    point = jnp.array([0.0, ellipsoid.generator_matrix[1, 1].astype(jnp.float64) * 0.5])
    tolerance = jnp.array(0.0, dtype=jnp.float64)

    assert ellipsoid.contains(point, tolerance=tolerance)
    assert jax.jit(
        lambda candidate_set, candidate_point, candidate_tolerance: (
            candidate_set.contains(candidate_point, tolerance=candidate_tolerance)
        )
    )(ellipsoid, point, tolerance)
