from .config import RenderConfig
from .mesh_projector import MeshWrapper
from .mesh_projector import read_mesh as _internal_read_mesh


class Mesh:
    """
    Represents a mesh object used by the mesh projector.

    This class provides a Pythonic wrapper around the internal mesh representation implemented in the Rust backend. It offers a convenient interface for performing mesh-related operations within the mesh_projector package.

    Users should not instantiate this class directly. Instead, instances should be created via the read_mesh function.

    This class is intended strictly for internal use within the mesh projector and is not part of the public API.
    """

    def __init__(self, mesh: MeshWrapper):
        self._internal_mesh = mesh


def read_mesh(obj_path: str, render_config: RenderConfig) -> Mesh:
    """
    Load a 3D mesh from disk, automatically inferring the optimal loading mode based on the provided files.

    This function supports three loading modes, selected automatically:
        - VertexColor: Loads mesh geometry and vertex colors (if present) from the OBJ file.
        - TextureMapped: Loads mesh geometry and UVs from the OBJ file, and applies the provided texture image.
        - BlinnPhong: Loads mesh geometry, UVs, and material properties from the OBJ and MTL files, and applies the texture image.

    The mesh is loaded into a format compatible with the Rust backend for efficient processing.

    Args:
        obj_path (str): Path to the OBJ file containing mesh geometry.
        render_config (Optional[RenderConfig], optional): The rendering configuration to use for the mesh (default: None).

    Returns:
        Mesh: A Mesh object ready for projection and point cloud operations.

    Raises:
        RuntimeError: If the mesh could not be loaded by the Rust backend.
    """
    try:
        internal_mesh = _internal_read_mesh(obj_path, render_config.model_dump())
        return Mesh(internal_mesh)
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during mesh reading: {e}")
