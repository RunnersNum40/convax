import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float

from convax import operations, sets


@jax.jit
def classify_robustly_admissible_states(
    output_limits: sets.HalfspacePolyhedron,
    output_disturbance: sets.Zonotope,
    state_limits: sets.HalfspacePolyhedron,
    output_matrix: Float[Array, "output_dimension state_dimension"],
    output_offset: Float[Array, "output_dimension"],
    candidate_states: Float[Array, "candidate state_dimension"],
) -> tuple[Bool[Array, "candidate"], Float[Array, "output_constraint"]]:
    if output_limits.equality_matrix.shape[0] != 0:
        raise ValueError("robust output limits must contain only inequalities")

    disturbance_margins = jax.vmap(output_disturbance.support_value)(
        output_limits.inequality_matrix
    )
    tightened_output_limits = sets.HalfspacePolyhedron(
        output_limits.inequality_matrix,
        output_limits.inequality_bounds - disturbance_margins,
    )
    robust_output_preimage = operations.affine_preimage(
        tightened_output_limits,
        output_matrix,
        output_offset,
    )
    admissible_states = operations.intersection(state_limits, robust_output_preimage)
    is_admissible = jax.vmap(admissible_states.contains)(candidate_states)
    return is_admissible, disturbance_margins


box_normals = jnp.array(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ]
)
output_limits = sets.HalfspacePolyhedron(
    box_normals,
    jnp.array([1.0, 1.0, 0.5, 0.5]),
)
output_disturbance = sets.Zonotope(
    center=jnp.zeros(2),
    generator_matrix=jnp.diag(jnp.array([0.1, 0.05])),
)
state_limits = sets.HalfspacePolyhedron(
    box_normals,
    jnp.array([2.0, 2.0, 1.0, 1.0]),
)
output_matrix = jnp.array([[1.0, 0.25], [0.0, 1.0]])
output_offset = jnp.array([0.1, 0.0])
candidate_states = jnp.array(
    [
        [0.0, 0.0],
        [0.7, 0.2],
        [0.9, 0.2],
        [-1.0, 0.0],
    ]
)

is_admissible, disturbance_margins = classify_robustly_admissible_states(
    output_limits,
    output_disturbance,
    state_limits,
    output_matrix,
    output_offset,
    candidate_states,
)

print("disturbance margins:", disturbance_margins)
print("candidate states:", candidate_states)
print("robustly admissible:", is_admissible)
