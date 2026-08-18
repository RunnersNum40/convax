import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from convax import AbstractSupportSet, Ellipsoid, affine_map


def support_values(
    convex_set: AbstractSupportSet,
    directions: Float[Array, "query ambient_dimension"],
) -> Float[Array, "query"]:
    return jax.vmap(convex_set.support_value)(directions)


def main() -> None:
    ellipsoid = Ellipsoid(
        center=jnp.array([1.0, -1.0]),
        generator_matrix=jnp.array([[2.0, 0.0], [0.0, 1.0]]),
    )
    transformed = affine_map(
        ellipsoid,
        matrix=jnp.array([[1.0, 0.5], [0.0, 2.0]]),
        offset=jnp.array([0.0, 1.0]),
    )
    directions = jnp.eye(2)
    compiled_support_values = jax.jit(support_values)
    print(compiled_support_values(transformed, directions))


if __name__ == "__main__":
    main()
