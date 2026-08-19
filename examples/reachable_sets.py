import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from convax import Ellipsoid, Zonotope, affine_map, convex_hull, minkowski_sum


@jax.jit
def query_reachable_envelope(
    initial_state: Ellipsoid,
    nominal_control: Zonotope,
    emergency_control: Zonotope,
    process_noise: Zonotope,
    directions: Float[Array, "query state_dimension"],
) -> tuple[
    Float[Array, "query"],
    Float[Array, "query state_dimension"],
    Float[Array, "state_dimension"],
    Float[Array, "state_dimension"],
]:
    time_step = 0.1
    dynamics_matrix = jnp.array([[1.0, time_step], [0.0, 1.0]])
    control_matrix = jnp.array([[0.5 * time_step**2], [time_step]])

    propagated_state = affine_map(initial_state, dynamics_matrix)
    nominal_reachable_set = minkowski_sum(
        propagated_state,
        affine_map(nominal_control, control_matrix),
    )
    emergency_reachable_set = minkowski_sum(
        propagated_state,
        affine_map(emergency_control, control_matrix),
    )
    reachable_envelope = minkowski_sum(
        convex_hull(nominal_reachable_set, emergency_reachable_set),
        process_noise,
    )

    support = jax.vmap(reachable_envelope.support)(directions)
    bounds = reachable_envelope.axis_aligned_bounds()
    return support.value, support.point, bounds.lower, bounds.upper


initial_state = Ellipsoid(
    center=jnp.array([0.0, 2.0]),
    generator_matrix=jnp.diag(jnp.array([0.1, 0.2])),
)
nominal_control = Zonotope(
    center=jnp.array([0.5]),
    generator_matrix=jnp.array([[0.1]]),
)
emergency_control = Zonotope(
    center=jnp.array([-1.5]),
    generator_matrix=jnp.array([[0.2]]),
)
process_noise = Zonotope(
    center=jnp.zeros(2),
    generator_matrix=jnp.diag(jnp.array([0.01, 0.05])),
)
directions = jnp.array([[1.0, 0.0], [0.0, 1.0]])

support_values, support_points, lower, upper = query_reachable_envelope(
    initial_state,
    nominal_control,
    emergency_control,
    process_noise,
    directions,
)

print("forward support values:", support_values)
print("forward support points:", support_points)
print("reachable state bounds:", lower, upper)
