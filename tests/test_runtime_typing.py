import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from convax import (
    AxisAlignedBounds,
    ConstrainedZonotope,
    HalfspacePolyhedron,
    Zonotope,
)


def test_array_constructor_axes_are_checked() -> None:
    with pytest.raises(TypeCheckError, match="parameter 'generator_matrix'"):
        Zonotope(jnp.zeros(2), jnp.zeros((3, 1)))

    with pytest.raises(TypeCheckError, match="parameter 'constraint_matrix'"):
        ConstrainedZonotope(
            jnp.zeros(2),
            jnp.zeros((2, 3)),
            jnp.zeros((1, 4)),
            jnp.zeros(1),
        )

    with pytest.raises(TypeCheckError, match="parameter 'equality_matrix'"):
        HalfspacePolyhedron(
            jnp.zeros((2, 3)),
            jnp.zeros(2),
            jnp.zeros((1, 4)),
            jnp.zeros(1),
        )

    with pytest.raises(TypeCheckError, match="parameter 'upper'"):
        AxisAlignedBounds(jnp.zeros(2), jnp.zeros(3))


def test_method_axes_are_checked() -> None:
    zonotope = Zonotope(jnp.zeros(2), jnp.eye(2))

    with pytest.raises(TypeCheckError, match="parameter 'direction'"):
        zonotope.support(jnp.zeros(3))

    with pytest.raises(TypeCheckError, match="parameter 'matrix'"):
        zonotope.affine_map(jnp.zeros((2, 3)))

    with pytest.raises(TypeCheckError, match="parameter 'offset'"):
        zonotope.affine_map(jnp.zeros((3, 2)), jnp.zeros(2))


def test_sequence_inputs_retain_manual_validation() -> None:
    zonotope = Zonotope([0, 0], [[1, 0], [0, 1]])

    assert zonotope.ambient_dimension == 2
    with pytest.raises(ValueError, match="rows must match"):
        Zonotope([0, 0], [[1], [2], [3]])


def test_runtime_type_checks_execute_during_jit_tracing() -> None:
    zonotope = Zonotope(jnp.zeros(2), jnp.eye(2))
    compiled_support = jax.jit(lambda direction: zonotope.support(direction))

    assert compiled_support(jnp.ones(2)).point.shape == (2,)
    with pytest.raises(TypeCheckError, match="parameter 'direction'"):
        compiled_support(jnp.ones(3))
