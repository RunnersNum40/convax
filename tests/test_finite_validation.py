from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from convax.operations import affine_map, affine_preimage, translate
from convax.sets import (
    AxisAlignedBounds,
    ConstrainedZonotope,
    Ellipsoid,
    HalfspacePolyhedron,
    SupportResult,
    VertexPolytope,
    Zonotope,
)


@pytest.mark.parametrize(
    ("construct", "parameter_name"),
    [
        pytest.param(
            lambda value: Ellipsoid(jnp.array([value]), [[1]]),
            "center",
            id="ellipsoid-center",
        ),
        pytest.param(
            lambda value: Ellipsoid([0], jnp.array([[value]])),
            "generator_matrix",
            id="ellipsoid-generator",
        ),
        pytest.param(
            lambda value: Zonotope(jnp.array([value]), [[1]]),
            "center",
            id="zonotope-center",
        ),
        pytest.param(
            lambda value: Zonotope([0], jnp.array([[value]])),
            "generator_matrix",
            id="zonotope-generator",
        ),
        pytest.param(
            lambda value: VertexPolytope(jnp.array([[value]])),
            "vertices",
            id="vertex-polytope",
        ),
        pytest.param(
            lambda value: ConstrainedZonotope(jnp.array([value]), [[1]], [[1]], [0]),
            "center",
            id="constrained-zonotope-center",
        ),
        pytest.param(
            lambda value: ConstrainedZonotope([0], jnp.array([[value]]), [[1]], [0]),
            "generator_matrix",
            id="constrained-zonotope-generator",
        ),
        pytest.param(
            lambda value: ConstrainedZonotope([0], [[1]], jnp.array([[value]]), [0]),
            "constraint_matrix",
            id="constrained-zonotope-constraint-matrix",
        ),
        pytest.param(
            lambda value: ConstrainedZonotope([0], [[1]], [[1]], jnp.array([value])),
            "constraint_values",
            id="constrained-zonotope-constraint-values",
        ),
        pytest.param(
            lambda value: HalfspacePolyhedron(jnp.array([[value]]), [1]),
            "inequality_matrix",
            id="halfspace-inequality-matrix",
        ),
        pytest.param(
            lambda value: HalfspacePolyhedron([[1]], jnp.array([value])),
            "inequality_bounds",
            id="halfspace-inequality-bounds",
        ),
        pytest.param(
            lambda value: HalfspacePolyhedron([[1]], [1], jnp.array([[value]]), [0]),
            "equality_matrix",
            id="halfspace-equality-matrix",
        ),
        pytest.param(
            lambda value: HalfspacePolyhedron([[1]], [1], [[1]], jnp.array([value])),
            "equality_values",
            id="halfspace-equality-values",
        ),
        pytest.param(
            lambda value: SupportResult(jnp.array(value), [0]),
            "value",
            id="support-result-value",
        ),
        pytest.param(
            lambda value: SupportResult(0, jnp.array([value])),
            "point",
            id="support-result-point",
        ),
        pytest.param(
            lambda value: AxisAlignedBounds(jnp.array([value]), [1]),
            "lower",
            id="axis-bounds-lower",
        ),
        pytest.param(
            lambda value: AxisAlignedBounds([-1], jnp.array([value])),
            "upper",
            id="axis-bounds-upper",
        ),
    ],
)
@pytest.mark.parametrize(
    "nonfinite_value",
    [
        pytest.param(jnp.nan, id="nan"),
        pytest.param(jnp.inf, id="positive-infinity"),
        pytest.param(-jnp.inf, id="negative-infinity"),
    ],
)
def test_public_constructors_reject_nonfinite_values(
    construct: Callable[[float], object],
    parameter_name: str,
    nonfinite_value: float,
) -> None:
    with pytest.raises(
        eqx.EquinoxRuntimeError,
        match=f"{parameter_name} must contain only finite values",
    ):
        construct(nonfinite_value)


@pytest.mark.parametrize(
    ("call", "parameter_name"),
    [
        pytest.param(
            lambda value: affine_map(Zonotope([0], [[1]]), jnp.array([[value]])),
            "matrix",
            id="affine-map-matrix",
        ),
        pytest.param(
            lambda value: affine_map(Zonotope([0], [[1]]), [[1]], jnp.array([value])),
            "offset",
            id="affine-map-offset",
        ),
        pytest.param(
            lambda value: affine_preimage(
                HalfspacePolyhedron([[1]], [1]), jnp.array([[value]])
            ),
            "matrix",
            id="affine-preimage-matrix",
        ),
        pytest.param(
            lambda value: affine_preimage(
                HalfspacePolyhedron([[1]], [1]), [[1]], jnp.array([value])
            ),
            "offset",
            id="affine-preimage-offset",
        ),
        pytest.param(
            lambda value: translate(
                HalfspacePolyhedron([[1]], [1]), jnp.array([value])
            ),
            "offset",
            id="halfspace-translation-offset",
        ),
        pytest.param(
            lambda value: Zonotope([0], [[1]]).support_value(jnp.array([value])),
            "direction",
            id="support-direction",
        ),
        pytest.param(
            lambda value: Ellipsoid([0], [[1]]).contains(jnp.array([value])),
            "point",
            id="ellipsoid-containment-point",
        ),
        pytest.param(
            lambda value: HalfspacePolyhedron(
                jnp.empty((0, 1)), jnp.empty((0,))
            ).contains(jnp.array([value])),
            "point",
            id="halfspace-containment-point",
        ),
    ],
)
@pytest.mark.parametrize(
    "nonfinite_value",
    [
        pytest.param(jnp.nan, id="nan"),
        pytest.param(jnp.inf, id="positive-infinity"),
        pytest.param(-jnp.inf, id="negative-infinity"),
    ],
)
def test_operations_reject_nonfinite_inputs(
    call: Callable[[float], object],
    parameter_name: str,
    nonfinite_value: float,
) -> None:
    with pytest.raises(
        eqx.EquinoxRuntimeError,
        match=f"{parameter_name} must contain only finite values",
    ):
        call(nonfinite_value)


@pytest.mark.parametrize(
    "nonfinite_value",
    [
        pytest.param(jnp.nan, id="nan"),
        pytest.param(jnp.inf, id="positive-infinity"),
        pytest.param(-jnp.inf, id="negative-infinity"),
    ],
)
def test_finite_validation_survives_jit_and_vmap(nonfinite_value: float) -> None:
    zonotope = Zonotope([0], [[1]])
    halfspace_polyhedron = HalfspacePolyhedron([[1]], [1])
    compiled_constructor = jax.jit(
        lambda center: Zonotope(center, jnp.ones((1, 1))).center
    )
    compiled_support_values = jax.jit(jax.vmap(zonotope.support_value))
    compiled_translation = jax.jit(
        lambda offset: translate(halfspace_polyhedron, offset)
    )

    with pytest.raises(jax.errors.JaxRuntimeError, match="center must contain"):
        compiled_constructor(jnp.array([nonfinite_value]))
    with pytest.raises(jax.errors.JaxRuntimeError, match="direction must contain"):
        compiled_support_values(jnp.array([[1.0], [nonfinite_value]]))
    with pytest.raises(jax.errors.JaxRuntimeError, match="offset must contain"):
        compiled_translation(jnp.array([nonfinite_value]))


def test_support_result_rejects_finite_input_overflow() -> None:
    largest_float = jnp.finfo(jnp.float32).max
    zonotope = Zonotope(
        jnp.zeros(1, dtype=jnp.float32),
        jnp.array([[largest_float]], dtype=jnp.float32),
    )
    direction = jnp.array([2.0], dtype=jnp.float32)

    with pytest.raises(eqx.EquinoxRuntimeError, match="value must contain"):
        zonotope.support(direction)
    with pytest.raises(jax.errors.JaxRuntimeError, match="value must contain"):
        jax.jit(lambda query: zonotope.support(query))(direction)


def test_derived_set_rejects_finite_input_overflow() -> None:
    largest_float = jnp.finfo(jnp.float32).max
    zonotope = Zonotope(
        jnp.array([largest_float], dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
    )
    matrix = jnp.array([[2.0]], dtype=jnp.float32)

    with pytest.raises(eqx.EquinoxRuntimeError, match="center must contain"):
        affine_map(zonotope, matrix)
    with pytest.raises(jax.errors.JaxRuntimeError, match="center must contain"):
        jax.jit(affine_map)(zonotope, matrix)


def test_axis_aligned_bounds_reject_reversed_bounds() -> None:
    with pytest.raises(eqx.EquinoxRuntimeError, match="must not exceed"):
        AxisAlignedBounds([1], [0])

    compiled_lower = jax.jit(lambda lower, upper: AxisAlignedBounds(lower, upper).lower)
    compiled_upper = jax.jit(lambda lower, upper: AxisAlignedBounds(lower, upper).upper)
    lower = jnp.array([1.0])
    upper = jnp.array([0.0])
    with pytest.raises(jax.errors.JaxRuntimeError, match="must not exceed"):
        compiled_lower(lower, upper)
    with pytest.raises(jax.errors.JaxRuntimeError, match="must not exceed"):
        compiled_upper(lower, upper)


def test_finite_validation_accepts_empty_arrays() -> None:
    empty_vector = jnp.empty((0,))
    empty_matrix = jnp.empty((0, 0))
    zonotope = Zonotope(empty_vector, empty_matrix)
    support = zonotope.support(empty_vector)
    bounds = AxisAlignedBounds(empty_vector, empty_vector)
    compiled_support = jax.jit(lambda query: zonotope.support(query))(empty_vector)
    compiled_bounds = jax.jit(AxisAlignedBounds)(empty_vector, empty_vector)

    assert support.value == 0
    assert support.point.shape == (0,)
    assert bounds.lower.shape == (0,)
    assert bounds.upper.shape == (0,)
    assert compiled_support.value == 0
    assert compiled_support.point.shape == (0,)
    assert compiled_bounds.lower.shape == (0,)
    assert compiled_bounds.upper.shape == (0,)
