import math
import os

import cv2
import mesh_projector as mp


def direction_to_view_angle(
    direction: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z = direction
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0.0:
        raise ValueError("Cannot convert a zero direction to a view angle")

    x /= length
    y /= length
    z /= length
    pitch = math.asin(max(-1.0, min(1.0, -y)))
    yaw = math.atan2(x, z)
    return (pitch, yaw, 0.0)


def icosahedron_face_normals() -> list[tuple[float, float, float]]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    directions = []
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            for z in (-1.0, 1.0):
                directions.append((x, y, z))
    for y in (-inv_phi, inv_phi):
        for z in (-phi, phi):
            directions.append((0.0, y, z))
    for x in (-inv_phi, inv_phi):
        for y in (-phi, phi):
            directions.append((x, y, 0.0))
    for x in (-phi, phi):
        for z in (-inv_phi, inv_phi):
            directions.append((x, 0.0, z))
    return directions


def directions_to_view_angles(
    directions: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    return [direction_to_view_angle(direction) for direction in directions]


if __name__ == "__main__":
    # Rendering configuration
    render_config = mp.RenderConfig(verbose=True)
    # Projection configuration
    projection_config = mp.ImageProjectionConfig(safety_area=False)
    # View angles
    view_angles = directions_to_view_angles(icosahedron_face_normals())
    # Read the mesh
    mesh = mp.read_mesh("assets/just_a_girl_C0-L5_deq_tri.obj", render_config)
    # Project to images
    projections, depths, normals, occupancies = mp.project_to_images(
        mesh, projection_config, view_angles
    )
    # Save the images
    os.makedirs("projections_icosahedron", exist_ok=True)
    for i, (p, d, n, o) in enumerate(zip(projections, depths, normals, occupancies)):
        p = cv2.cvtColor(p, cv2.COLOR_RGBA2BGRA)
        d = cv2.cvtColor(d, cv2.COLOR_RGBA2BGRA)
        n = cv2.cvtColor(n, cv2.COLOR_RGBA2BGRA)
        o = cv2.cvtColor(o, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"projections_icosahedron/projection_{i}.png", p)
        cv2.imwrite(f"projections_icosahedron/depth_map_{i}.png", d)
        cv2.imwrite(f"projections_icosahedron/normal_map_{i}.png", n)
        cv2.imwrite(f"projections_icosahedron/occupancy_map_{i}.png", o)
