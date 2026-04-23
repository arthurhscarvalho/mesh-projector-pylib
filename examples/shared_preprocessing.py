import math

import mesh_projector as mp


if __name__ == "__main__":
    render_config = mp.RenderConfig(verbose=True, use_gpu_renderer=True, resolution=512)
    projection_config = mp.ImageProjectionConfig()
    pi = math.pi
    view_angles = [
        (0.0, 0.0, 0.0),
        (0.0, pi, 0.0),
        (0.0, pi / 2.0, 0.0),
        (0.0, -pi / 2.0, 0.0),
        (pi / 2.0, 0.0, 0.0),
        (-pi / 2.0, 0.0, 0.0),
    ]

    reference_mesh = mp.read_mesh("assets/just_a_girl_C0-L5_deq_tri.obj", render_config)
    shared_preprocessing = mp.build_preprocessing(reference_mesh, render_config.resolution)

    distorted_mesh = mp.read_mesh("assets/just_a_girl_C0-L5_deq_tri.obj", render_config)

    reference_ready = mp.apply_preprocessing(reference_mesh, shared_preprocessing)
    distorted_ready = mp.apply_preprocessing(distorted_mesh, shared_preprocessing)

    reference_images, *_ = mp.project_to_images(
        reference_ready, projection_config, view_angles
    )
    distorted_images, *_ = mp.project_to_images(
        distorted_ready, projection_config, view_angles
    )

    print(reference_images.shape, distorted_images.shape)
