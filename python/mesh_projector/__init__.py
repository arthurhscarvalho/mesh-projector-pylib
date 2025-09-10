from typing import List, Optional, Tuple

import numpy as np

from .config import (
    AmbientLight,
    DirectionalLight,
    Headlight,
    LightConfig,
    PointLights,
    RenderConfig,
)
from .mesh_projector import MeshWrapper
from .mesh_projector import project_to_images as _internal_project_to_images
from .mesh_projector import project_to_point_cloud as _internal_project_to_point_cloud
from .mesh_projector import read_mesh as _internal_read_mesh


class Mesh:
    """
    Represents a mesh object for use with the mesh projector.

    This class serves as a Pythonic wrapper around the internal mesh representation used by the Rust backend.
    It is intended to provide a convenient interface for mesh operations within the mesh projector package.
    Users should not instantiate this class directly; instead, obtain instances via the `read_mesh` function.
    This class is intended for internal use within the mesh projector and is not designed for use outside this context.
    """

    def __init__(self, mesh: MeshWrapper):
        self._internal_mesh = mesh


def read_mesh(
    obj_path: str,
    render_config: Optional[RenderConfig] = None,
    light_config: Optional[LightConfig] = None,
) -> Mesh:
    """
    Load a 3D mesh from disk, automatically inferring the optimal loading mode based on the provided files.

    This function supports three loading modes, selected automatically:
        - VertexColor: Loads mesh geometry and vertex colors (if present) from the OBJ file.
        - TextureMapped: Loads mesh geometry and UVs from the OBJ file, and applies the provided texture image.
        - BlinnPhong: Loads mesh geometry, UVs, and material properties from the OBJ and MTL files, and applies the texture image.

    The mesh is loaded into a format compatible with the Rust backend for efficient processing.

    Args:
        obj_path (str): Path to the OBJ file containing mesh geometry.
        shading_mode (Optional[Literal["vertex_color", "texture_mapped", "blinn_phong"]], optional): Mesh mode to use for loading the mesh (default: None).
        resolution (int, optional): Resolution for mesh normalization and rasterization (default: 4096).
            The mesh will be scaled to the resolution.
            The default resolution is 4096, which results in 4096x4096 resolution for the bounding cube.
        flip_uv (bool, optional): Whether to flip the V coordinate of UVs (default: True).
        ignore_opacity (bool, optional): If True, ignore opacity/alpha in materials and textures (default: False).
        verbose (bool, optional): If True, enable verbose logging from the Rust backend (default: False).

    Returns:
        Mesh: A Mesh object ready for projection and point cloud operations.

    Raises:
        RuntimeError: If the mesh could not be loaded by the Rust backend.
    """
    if render_config is None:
        render_config = RenderConfig()
    if light_config is None:
        # Default to a camera-attached headlight
        light_config = Headlight()
    try:
        internal_mesh = _internal_read_mesh(
            obj_path,
            render_config.model_dump(),
            light_config.model_dump(),
        )
        return Mesh(internal_mesh)
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during mesh reading: {e}")


def project_to_images(
    mesh: Mesh,
    view_angles: List[Tuple[float, float, float]],
    filtering_radius: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 6 orthographic projections of the mesh, one for each face of the bounding cube.

    Args:
        mesh (Mesh): The mesh object to project (as returned by `read_mesh`).
        view_angles (List[Tuple[float, float, float]]): List of view angles for the projections.
        filtering_radius (int, optional): Radius for outlier filtering in the depth map (default: 0, disables filtering).
        verbose (bool, optional): If True, enables verbose logging from the Rust backend (default: False).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - projections: (6, H, W, 3) uint8 RGB images for each cube face
            - depth_maps: (6, H, W) uint8 depth maps for each face
            - normal_maps: (6, H, W, 3) uint8 normal maps for each face
            - occlusion_maps: (6, H, W) uint8 occupancy maps for each face (255=mesh, 0=background)

    Raises:
        RuntimeError: If the Rust backend fails during projection generation.
    """
    try:
        return _internal_project_to_images(
            mesh._internal_mesh,
            view_angles,
            filtering_radius,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during projection generation: {e}")


def project_to_point_cloud(
    mesh: Mesh,
    density_factor: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a point cloud from the mesh by sampling its surface, returning the 3D points and their corresponding colors.

    Args:
        mesh (Mesh): The mesh object to sample from (as returned by `read_mesh`).
        density_factor (float, optional): Controls the density of the sampled point cloud. Higher values produce more points (default: 1.0).
        verbose (bool, optional): If True, enables verbose logging from the Rust backend (default: False).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - points (np.ndarray): Array of shape (N, 3) containing the 3D coordinates (f32) of the sampled points.
            - colors (np.ndarray): Array of shape (N, 3) containing the RGB color values (uint8) for each point.

    Raises:
        RuntimeError: If the Rust backend fails during point cloud generation.
    """
    try:
        return _internal_project_to_point_cloud(
            mesh._internal_mesh,
            density_factor,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during point cloud generation: {e}")


__all__ = [
    "Mesh",
    "read_mesh",
    "project_to_images",
    "project_to_point_cloud",
    "RenderConfig",
    "LightConfig",
    "Headlight",
    "AmbientLight",
    "DirectionalLight",
    "PointLights",
]
