import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import cv2
import mesh_projector as mp
import numpy as np


if __name__ == "__main__":
    # Rendering configuration
    render_config = mp.RenderConfig(resolution=1024, verbose=False)
    # Projection configuration
    projection_config = mp.ImageProjectionConfig(
        verbose=False,
        camera_config=mp.PerspectiveCamera(vertical_fov_degrees=60),
        msaa_config=mp.MsaaConfig(sample_count=mp.MsaaSampleCount.X8),
        ssao_config=mp.SsaoConfig(enabled=True),
        background_config=mp.BackgroundConfig(alpha=255),
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
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(mp.project_to_images, mesh, projection_config, [view_angle])
            for view_angle in view_angles
        ]
        future_to_index = {future: index for index, future in enumerate(futures)}
        # Save the images
        os.makedirs("projections", exist_ok=True)
        pbar = tqdm(total=len(futures), desc="Commputing projections", colour="magenta")
        for future in as_completed(futures):
            # Get future index
            idx = future_to_index[future]
            # Check for possible exceptions
            if future.exception():
                raise future.exception()
            # We will save only the projestion (the rendered mesh)
            result = future.result()
            # Remove the first dimension, since it's only 1 image
            projection = np.squeeze(result[0], axis=0)
            # Convert from RGBA to BGRA (for cv2)
            projection = cv2.cvtColor(projection, cv2.COLOR_RGBA2BGRA)
            # Save image and update progress bar
            cv2.imwrite(f"projections/projection_{idx:04}.png", projection)
            pbar.update(n=1)
        pbar.close()
