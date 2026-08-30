import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from convax.sets import (
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
    with pytest.raises(TypeCheckError, match="parameter 'tolerance'"):
        containment_set.contains([0], tolerance=jnp.array([1e-3]))


def test_containment_rejects_complex_tolerance(
    containment_set: AbstractPointContainmentSet,
) -> None:
    with pytest.raises(TypeError, match="requires real-valued arrays"):
        containment_set.contains([0], tolerance=jnp.array(1e-3 + 1e-3j))


@pytest.mark.parametrize(
    ("tolerance", "message"),
    [
        pytest.param(-1e-3, "must be nonnegative", id="negative"),
        pytest.param(jnp.inf, "must contain only finite", id="positive-infinity"),
        pytest.param(-jnp.inf, "must contain only finite", id="negative-infinity"),
        pytest.param(jnp.nan, "must contain only finite", id="nan"),
    ],
)
def test_containment_rejects_invalid_tolerance(
    containment_set: AbstractPointContainmentSet,
    tolerance: float | jax.Array,
    message: str,
) -> None:
    compiled_contains = jax.jit(
        lambda point, candidate_tolerance: containment_set.contains(
            point, tolerance=candidate_tolerance
        )
    )

    with pytest.raises(eqx.EquinoxRuntimeError, match=message):
        containment_set.contains([0], tolerance=tolerance)
    with pytest.raises(jax.errors.JaxRuntimeError, match=message):
        compiled_contains(jnp.zeros(1), tolerance)


def test_ellipsoid_containment_preserves_ill_conditioned_full_rank_axis() -> None:
    ellipsoid = Ellipsoid(
        jnp.zeros(2, dtype=jnp.float32),
        jnp.diag(jnp.array([1e6, 1.0], dtype=jnp.float32)),
    )
    point = jnp.array([0.0, 0.5], dtype=jnp.float32)

    assert ellipsoid.contains(point)
    assert jax.jit(lambda candidate_point: ellipsoid.contains(candidate_point))(point)


def test_ellipsoid_containment_accepts_relative_rank_tolerance() -> None:
    ellipsoid = Ellipsoid(
        jnp.zeros(2, dtype=jnp.float32),
        jnp.diag(jnp.array([1e6, 1.0], dtype=jnp.float32)),
    )
    point = jnp.array([0.0, 0.5], dtype=jnp.float32)
    relative_rank_tolerance = jnp.array(2e-6, dtype=jnp.float32)

    assert not ellipsoid.contains(
        point,
        relative_rank_tolerance=relative_rank_tolerance,
    )
    assert not jax.jit(
        lambda candidate_rank_tolerance: ellipsoid.contains(
            point,
            relative_rank_tolerance=candidate_rank_tolerance,
        )
    )(relative_rank_tolerance)


def test_ellipsoid_containment_rejects_vector_relative_rank_tolerance() -> None:
    ellipsoid = Ellipsoid([0], [[1]])

    with pytest.raises(TypeCheckError, match="parameter 'relative_rank_tolerance'"):
        ellipsoid.contains([0], relative_rank_tolerance=jnp.array([1e-6]))


def test_ellipsoid_containment_rejects_complex_relative_rank_tolerance() -> None:
    ellipsoid = Ellipsoid([0], [[1]])

    with pytest.raises(TypeError, match="requires real-valued arrays"):
        ellipsoid.contains(
            [0],
            relative_rank_tolerance=jnp.array(1e-6 + 1e-6j),
        )


@pytest.mark.parametrize(
    ("relative_rank_tolerance", "message"),
    [
        pytest.param(-1e-3, "must be nonnegative", id="negative"),
        pytest.param(jnp.inf, "must contain only finite", id="positive-infinity"),
        pytest.param(-jnp.inf, "must contain only finite", id="negative-infinity"),
        pytest.param(jnp.nan, "must contain only finite", id="nan"),
    ],
)
def test_ellipsoid_containment_rejects_invalid_relative_rank_tolerance(
    relative_rank_tolerance: float | jax.Array,
    message: str,
) -> None:
    ellipsoid = Ellipsoid([0], [[1]])
    compiled_contains = jax.jit(
        lambda candidate_rank_tolerance: ellipsoid.contains(
            [0],
            relative_rank_tolerance=candidate_rank_tolerance,
        )
    )

    with pytest.raises(eqx.EquinoxRuntimeError, match=message):
        ellipsoid.contains(
            [0],
            relative_rank_tolerance=relative_rank_tolerance,
        )
    with pytest.raises(jax.errors.JaxRuntimeError, match=message):
        compiled_contains(relative_rank_tolerance)


@pytest.mark.parametrize(
    "dtype", [jnp.float16, jnp.bfloat16], ids=["float16", "bfloat16"]
)
def test_ellipsoid_containment_promotes_low_precision_pseudoinverse(
    dtype: jnp.dtype,
) -> None:
    ellipsoid = Ellipsoid(jnp.zeros(2, dtype=dtype), jnp.eye(2, dtype=dtype))
    points = jnp.array([[0.5, 0.0], [2.0, 0.0]], dtype=dtype)
    tolerance = jnp.array(0.0, dtype=dtype)
    expected = jnp.array([True, False])

    eager = jax.vmap(lambda point: ellipsoid.contains(point, tolerance=tolerance))(
        points
    )
    compiled = jax.jit(
        jax.vmap(lambda point: ellipsoid.contains(point, tolerance=tolerance))
    )(points)

    assert jnp.array_equal(eager, expected)
    assert jnp.array_equal(compiled, expected)


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
def test_containment_preserves_small_negative_tolerance_during_dtype_promotion(
    containment_set: AbstractPointContainmentSet,
) -> None:
    tolerance = jnp.array(-1e-8, dtype=jnp.float32)

    with pytest.raises(eqx.EquinoxRuntimeError, match="must be nonnegative"):
        containment_set.contains([0], tolerance=tolerance)

    compiled_contains = jax.jit(
        lambda candidate_tolerance: containment_set.contains(
            [0], tolerance=candidate_tolerance
        )
    )
    with pytest.raises(jax.errors.JaxRuntimeError, match="must be nonnegative"):
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
