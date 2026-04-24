import math
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import mesh_projector as mp
import numpy as np
from tqdm import tqdm


def project_and_save_image(index_and_view_angle):
    idx, view_angle = index_and_view_angle
    projections, depths, normals, occupancies = mp.project_to_images(
        mesh, projection_config, [view_angle]
    )
    # Keep zero-copy readback arrays thread-local; only return plain metadata.
    projection = np.squeeze(projections, axis=0)
    projection = cv2.cvtColor(projection, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(f"projections/projection_{idx:04}.png", projection)
    return idx


if __name__ == "__main__":
    # Rendering configuration
    render_config = mp.RenderConfig(
        resolution=1920,
        verbose=False,
        use_gpu_renderer=True,
    )
    # Projection configuration
    projection_config = mp.ImageProjectionConfig(
        verbose=False,
        camera_config=mp.PerspectiveCamera(vertical_fov_degrees=60),
        msaa_config=mp.MsaaConfig(sample_count=mp.MsaaSampleCount.X8),
        ssao_config=mp.SsaoConfig(enabled=True),
        background_config=mp.BackgroundConfig(color=(1.0, 1.0, 1.0, 1.0)),
        light_config=mp.PointLights(positions=[(-1.0, 1.0, 1.0), (1.0, -1.0, -1.0)]),
    )
    # View angles
    PI = math.pi
    view_angles = (
        [(0.0, 2 * PI * i / 240, 0.0) for i in range(240)]  # One full loop in 4 s
        + [(0.0, 0.0, 2 * PI * i / 240) for i in range(240)]
        + [(2 * PI * i / 240, 0.0, 0.0) for i in range(240)]
    )
    # Read the mesh
    mesh = mp.read_mesh("assets/just_a_girl_C0-L5_deq_tri.obj", render_config)
    preprocessing = mp.build_preprocessing(mesh, render_config)
    mesh = mp.apply_preprocessing(mesh, preprocessing)
    # Project to images
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Save the images
        os.makedirs("projections", exist_ok=True)
        pbar = tqdm(total=len(view_angles), desc="Computing projections")
        for _ in executor.map(project_and_save_image, enumerate(view_angles)):
            pbar.update(n=1)
        pbar.close()
