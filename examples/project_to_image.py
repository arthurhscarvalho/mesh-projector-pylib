import math
import os

import cv2
import mesh_projector as mp

if __name__ == "__main__":
    # Rendering configuration
    render_config = mp.RenderConfig(
        shading_mode="blinn_phong",
        resolution=4096,
        flip_uv=True,
        ignore_opacity=False,
        verbose=True,
    )
    # Lighting configuration
    light_config = mp.PointLight(
        position=(0.0, 1.0, 1.0),
        global_ambient=(0.5, 0.5, 0.5),
        global_diffuse=(1.0, 1.0, 1.0),
        global_specular=(0.0, 0.0, 0.0),
    )
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
    mesh = mp.read_mesh(
        "assets/just_a_girl_C0-L5_deq_tri.obj",
        render_config,
        light_config,
    )
    # Project to images
    projections, depths, normals, occlusions = mp.project_to_images(mesh, view_angles)
    # Save the images
    os.makedirs("projections", exist_ok=True)
    for i, (p, d, n, o) in enumerate(zip(projections, depths, normals, occlusions)):
        p = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
        d = cv2.cvtColor(d, cv2.COLOR_RGB2BGR)
        n = cv2.cvtColor(n, cv2.COLOR_RGB2BGR)
        o = cv2.cvtColor(o, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"projections/projection_{i}.png", p)
        cv2.imwrite(f"projections/depth_map_{i}.png", d)
        cv2.imwrite(f"projections/normal_map_{i}.png", n)
        cv2.imwrite(f"projections/occlusion_map_{i}.png", o)
