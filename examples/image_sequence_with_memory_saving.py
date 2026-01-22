import math
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import cv2
import mesh_projector as mp
import numpy as np


if __name__ == "__main__":
    # Rendering configuration
    render_config = mp.RenderConfig(resolution=1920, verbose=False)
    # Projection configuration
    projection_config = mp.ImageProjectionConfig(
        verbose=False,
        camera_config=mp.PerspectiveCamera(vertical_fov_degrees=60),
        msaa_config=mp.MsaaConfig(sample_count=mp.MsaaSampleCount.X8),
        ssao_config=mp.SsaoConfig(enabled=True),
        background_config=mp.BackgroundConfig(color=(1.0, 1.0, 1.0)),
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
    # Project to images
    with ThreadPoolExecutor() as executor:
        # Save the images
        os.makedirs("projections", exist_ok=True)
        pbar = tqdm(total=len(view_angles), desc="Commputing projections")
        for idx, result in enumerate(
            executor.map(
                lambda va: mp.project_to_images(mesh, projection_config, [va]),
                view_angles,
            )
        ):
            # Remove the first dimension, since it's only 1 image
            projection = np.squeeze(result[0], axis=0)
            # Convert from RGBA to BGRA (for cv2)
            projection = cv2.cvtColor(projection, cv2.COLOR_RGBA2BGRA)
            # Save image and update progress bar
            cv2.imwrite(f"projections/projection_{idx:04}.png", projection)
            pbar.update(n=1)
        pbar.close()
