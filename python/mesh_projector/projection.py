from typing import List, Optional, Tuple

from numpy.typing import NDArray

from .config import ImageProjectionConfig, PointCloudProjectionConfig
from .mesh import Mesh, MeshPreprocessing
from .mesh_projector import project_to_images as _internal_project_to_images
from .mesh_projector import project_to_point_cloud as _internal_project_to_point_cloud


def project_to_images(
    mesh: Mesh,
    projection_config: ImageProjectionConfig,
    view_angles: List[Tuple[float, float, float]],
    preprocessing: Optional[MeshPreprocessing] = None,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Generate 6 projections of the mesh (orthographic or perspective), one for each face of the bounding cube.

    Args:
        mesh (Mesh): The mesh object to project (as returned by `read_mesh`).
        view_angles (List[Tuple[float, float, float]]): List of view angles for the projections.
        projection_config (ImageProjectionConfig): Bundled projection config including filtering, lighting, camera, MSAA, SSAO, background, and verbosity.
        preprocessing (Optional[MeshPreprocessing]): Optional preprocessing recipe to apply temporarily during projection.

    Returns:
        Tuple[NDArray, NDArray, NDArray, NDArray]:
            - projections: (6, H, W, 4) read-only uint8 RGBA images for each cube face
            - depth_maps: (6, H, W) read-only uint8 depth maps for each face
            - normal_maps: (6, H, W, 4) read-only uint8 normal maps for each face; use normal_maps[..., :3] for XYZ/RGB data
            - occlusion_maps: (6, H, W) read-only uint8 occupancy maps for each face (255=mesh, 0=background)

    Raises:
        RuntimeError: If the Rust backend fails during projection generation.
    """
    try:
        return _internal_project_to_images(
            mesh._internal_mesh,
            projection_config.model_dump(),
            view_angles,
            None if not preprocessing else preprocessing._internal_preprocessing,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during projection generation: {e}")


def project_to_point_cloud(
    mesh: Mesh,
    projection_config: PointCloudProjectionConfig,
    preprocessing: Optional[MeshPreprocessing] = None,
) -> Tuple[NDArray, NDArray]:
    """
    Generate a point cloud from the mesh by sampling its surface, returning the 3D points and their corresponding colors.

    Args:
        mesh (Mesh): The mesh object to sample from (as returned by `read_mesh`).
        projection_config (PointCloudProjectionConfig): Bundled config containing lighting, density factor, and verbosity for point cloud generation.
        preprocessing (Optional[MeshPreprocessing]): Optional preprocessing recipe to apply temporarily during point-cloud generation.

    Returns:
        Tuple[NDArray, NDArray]:
            - points (NDArray): Array of shape (N, 3) containing the 3D coordinates (f32) of the sampled points.
            - colors (NDArray): Array of shape (N, 3) containing the RGB color values (uint8) for each point.

    Raises:
        RuntimeError: If the Rust backend fails during point cloud generation.
    """
    try:
        return _internal_project_to_point_cloud(
            mesh._internal_mesh,
            projection_config.model_dump(),
            None if preprocessing is None else preprocessing._internal_preprocessing,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during point cloud generation: {e}")
