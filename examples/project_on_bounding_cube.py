import math
import os

import cv2
import mesh_projector as mp

if __name__ == "__main__":
    # Rendering configuration
    render_config = mp.RenderConfig(verbose=True, use_gpu_renderer=True)
    # Projection configuration
    projection_config = mp.ImageProjectionConfig()
    # View angles
    PI = math.pi
    view_angles = [
        (0.0, 0.0, 0.0),  # Front: Cam at +Z, looking -Z.
        (0.0, PI, 0.0),  # Back: Cam at -Z, looking +Z.
        (0.0, PI / 2.0, 0.0),  # Right: Cam at +X, looking -X.
        (0.0, -PI / 2.0, 0.0),  # Left: Cam at -X, looking +X.
        (PI / 2.0, 0.0, 0.0),  # Top: Cam at +Y, looking -Y.
        (-PI / 2.0, 0.0, 0.0),  # Bottom: Cam at -Y, looking +Y.
    ]
    # Read the mesh
    mesh = mp.read_mesh("assets/just_a_girl_C0-L5_deq_tri.obj", render_config)
    preprocessing = mp.build_preprocessing(mesh, render_config)
    mesh = mp.apply_preprocessing(mesh, preprocessing)
    # Project to images
    projections, depths, normals, occupancies = mp.project_to_images(
        mesh, projection_config, view_angles
    )
    # Save the images
    os.makedirs("projections", exist_ok=True)
    for i, (p, d, n, o) in enumerate(zip(projections, depths, normals, occupancies)):
        p = cv2.cvtColor(p, cv2.COLOR_RGBA2BGRA)
        d = cv2.cvtColor(d, cv2.COLOR_RGBA2BGRA)
        n = cv2.cvtColor(n, cv2.COLOR_RGBA2BGRA)
        o = cv2.cvtColor(o, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"projections/projection_{i}.png", p)
        cv2.imwrite(f"projections/depth_map_{i}.png", d)
        cv2.imwrite(f"projections/normal_map_{i}.png", n)
        cv2.imwrite(f"projections/occupancy_map_{i}.png", o)
