import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from convax import sets


@jax.jit
def query_support_boundary(
    convex_set: sets.AbstractSupportSet,
    directions: Float[Array, "query ambient_dimension"],
) -> tuple[
    Float[Array, "query"],
    Float[Array, "query ambient_dimension"],
    Float[Array, "ambient_dimension"],
    Float[Array, "ambient_dimension"],
]:
    support = jax.vmap(convex_set.support)(directions)
    bounds = convex_set.axis_aligned_bounds()
    return support.value, support.point, bounds.lower, bounds.upper


reachable_position = sets.Zonotope(
    center=jnp.array([1.0, -0.5]),
    generator_matrix=jnp.array(
        [
            [1.2, 0.4, 0.0],
            [0.1, 0.3, 0.8],
        ]
    ),
)
angles = jnp.linspace(0.0, 2.0 * jnp.pi, num=8, endpoint=False)
directions = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=1)

support_values, support_points, lower, upper = query_support_boundary(
    reachable_position, directions
)

print("support values:", support_values)
print("support points:", support_points)
print("axis-aligned bounds:", lower, upper)
