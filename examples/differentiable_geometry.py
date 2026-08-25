import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from convax import Ellipsoid, affine_map


def collision_clearance(
    pose: Float[Array, "3"],
    local_footprint: Ellipsoid,
    obstacle_normal: Float[Array, "2"],
    obstacle_bound: Float[Array, ""],
) -> Float[Array, ""]:
    cosine = jnp.cos(pose[2])
    sine = jnp.sin(pose[2])
    rotation_matrix = jnp.array([[cosine, -sine], [sine, cosine]])
    world_footprint = affine_map(local_footprint, rotation_matrix, pose[:2])
    return obstacle_bound - world_footprint.support_value(obstacle_normal)


def evaluate_clearance_ascent_step(
    pose: Float[Array, "3"],
    local_footprint: Ellipsoid,
    obstacle_normal: Float[Array, "2"],
    obstacle_bound: Float[Array, ""],
) -> tuple[
    Float[Array, ""],
    Float[Array, "3"],
    Float[Array, "3"],
    Float[Array, ""],
]:
    clearance, clearance_gradient = jax.value_and_grad(collision_clearance)(
        pose, local_footprint, obstacle_normal, obstacle_bound
    )
    improved_pose = pose + 0.1 * clearance_gradient
    improved_clearance = collision_clearance(
        improved_pose,
        local_footprint,
        obstacle_normal,
        obstacle_bound,
    )
    return clearance, clearance_gradient, improved_pose, improved_clearance


local_footprint = Ellipsoid(
    center=jnp.zeros(2),
    generator_matrix=jnp.diag(jnp.array([0.6, 0.25])),
)
pose = jnp.array([0.5, -0.2, 0.4])
obstacle_normal = jnp.array([1.0, 0.5])
obstacle_normal = obstacle_normal / jnp.linalg.norm(obstacle_normal)
obstacle_bound = jnp.array(2.0)

clearance, clearance_gradient, improved_pose, improved_clearance = jax.jit(
    evaluate_clearance_ascent_step
)(pose, local_footprint, obstacle_normal, obstacle_bound)

print("clearance:", clearance)
print("clearance gradient [x, y, heading]:", clearance_gradient)
print("improved pose:", improved_pose)
print("improved clearance:", improved_clearance)
