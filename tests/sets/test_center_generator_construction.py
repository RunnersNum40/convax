from typing import cast

import jax.numpy as jnp
import pytest

from convax import Ellipsoid, Zonotope
from convax._utils import MatrixLike, VectorLike

type CenterGeneratorSet = type[Ellipsoid] | type[Zonotope]


@pytest.mark.parametrize("set_type", [Ellipsoid, Zonotope])
def test_integer_lists_become_floating_arrays(
    set_type: CenterGeneratorSet,
) -> None:
    convex_set = set_type([1, -2], [[3], [4]])
    expected_dtype = jnp.result_type(0.0)

    assert convex_set.center.dtype == expected_dtype
    assert convex_set.generator_matrix.dtype == expected_dtype
    assert jnp.array_equal(convex_set.center, jnp.array([1.0, -2.0]))
    assert jnp.array_equal(convex_set.generator_matrix, jnp.array([[3.0], [4.0]]))


@pytest.mark.parametrize("set_type", [Ellipsoid, Zonotope])
def test_mixed_floating_dtypes_follow_result_type(
    set_type: CenterGeneratorSet,
) -> None:
    center = jnp.array([1.0, -2.0], dtype=jnp.float16)
    generator_matrix = jnp.array(
        [[3.0], [4.0]],
        dtype=jnp.asarray(0.0).dtype,
    )
    expected_dtype = jnp.result_type(center.dtype, generator_matrix.dtype)

    convex_set = set_type(center, generator_matrix)

    assert convex_set.center.dtype == expected_dtype
    assert convex_set.generator_matrix.dtype == expected_dtype


@pytest.mark.parametrize("set_type", [Ellipsoid, Zonotope])
def test_constructor_preserves_shape_validation_errors(
    set_type: CenterGeneratorSet,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^center must be a vector, got shape \(1, 2\)$",
    ):
        set_type(cast(VectorLike, [[1.0, 2.0]]), [[1.0], [2.0]])

    with pytest.raises(
        ValueError,
        match=r"^generator_matrix must be a matrix, got shape \(3,\)$",
    ):
        set_type([1.0, 2.0], cast(MatrixLike, [1.0, 2.0, 3.0]))

    with pytest.raises(
        ValueError,
        match=(
            r"^generator_matrix rows must match the center dimension, "
            r"got \(3, 1\) and \(2,\)$"
        ),
    ):
        set_type([1.0, 2.0], [[1.0], [2.0], [3.0]])


@pytest.mark.parametrize("set_type", [Ellipsoid, Zonotope])
def test_constructor_rejects_complex_inputs(
    set_type: CenterGeneratorSet,
) -> None:
    message = "^Convax requires real-valued arrays$"

    with pytest.raises(TypeError, match=message):
        set_type(jnp.array([1.0 + 1.0j]), [[1.0]])

    with pytest.raises(TypeError, match=message):
        set_type([1.0], jnp.array([[1.0 + 1.0j]]))


@pytest.mark.parametrize("set_type", [Ellipsoid, Zonotope])
def test_constructor_preserves_conversion_and_validation_order(
    set_type: CenterGeneratorSet,
) -> None:
    with pytest.raises(
        TypeError,
        match=r"^Convax requires real-valued arrays$",
    ):
        set_type(cast(VectorLike, [[1.0]]), jnp.array([[1.0 + 1.0j]]))

    with pytest.raises(
        ValueError,
        match=r"^center must be a vector, got shape \(1, 1\)$",
    ):
        set_type(cast(VectorLike, [[1.0]]), cast(MatrixLike, [1.0]))
