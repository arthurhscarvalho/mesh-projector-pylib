from .config import RenderConfig
from .mesh_projector import MeshPreprocessingWrapper, MeshWrapper
from .mesh_projector import apply_preprocessing as _internal_apply_preprocessing
from .mesh_projector import build_preprocessing as _internal_build_preprocessing
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

    def get_transform(self) -> "MeshPreprocessing":
        return MeshPreprocessing(self._internal_mesh.get_transform())


class MeshPreprocessing:
    """Reusable geometry preprocessing recipe for meshes."""

    def __init__(self, preprocessing: MeshPreprocessingWrapper):
        self._internal_preprocessing = preprocessing

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple(self._internal_preprocessing.center)

    @property
    def scale(self) -> float:
        return self._internal_preprocessing.scale

    @property
    def shift(self) -> tuple[float, float, float]:
        return tuple(self._internal_preprocessing.shift)

    @property
    def target_max_val(self) -> float:
        return self._internal_preprocessing.target_max_val

    @property
    def clamp_to_target_bounds(self) -> bool:
        return self._internal_preprocessing.clamp_to_target_bounds

    def __repr__(self) -> str:
        return repr(self._internal_preprocessing)


def read_mesh(obj_path: str, render_config: RenderConfig) -> Mesh:
    """
    Load a raw 3D mesh from disk, automatically inferring the optimal loading mode based on the provided files.

    This function supports three loading modes, selected automatically:
        - VertexColor: Loads mesh geometry and vertex colors (if present) from the OBJ file.
        - TextureMapped: Loads mesh geometry and UVs from the OBJ file, and applies the provided texture image.
        - BlinnPhong: Loads mesh geometry, UVs, and material properties from the OBJ and MTL files, and applies the texture image.

    The mesh is loaded into a format compatible with the Rust backend for efficient processing.
    Geometry preprocessing is built separately and can be applied explicitly with
    ``build_preprocessing`` and ``apply_preprocessing``.

    Args:
        obj_path (str): Path to the OBJ file containing mesh geometry.
        render_config (Optional[RenderConfig], optional): The rendering configuration to use for the mesh (default: None).

    Returns:
        Mesh: A Mesh object containing the raw scene plus its default preprocessing recipe.

    Raises:
        RuntimeError: If the mesh could not be loaded by the Rust backend.
    """
    try:
        internal_mesh = _internal_read_mesh(obj_path, render_config.model_dump())
        return Mesh(internal_mesh)
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during mesh reading: {e}")


def build_preprocessing(mesh: Mesh, render_config: RenderConfig) -> MeshPreprocessing:
    """Build a reusable preprocessing recipe from a mesh."""
    try:
        preprocessing = _internal_build_preprocessing(
            mesh._internal_mesh, render_config.resolution
        )
        return MeshPreprocessing(preprocessing)
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during preprocessing build: {e}")


def apply_preprocessing(mesh: Mesh, preprocessing: MeshPreprocessing) -> Mesh:
    """Apply preprocessing to a mesh and return a new projection-ready mesh."""
    try:
        processed_mesh = _internal_apply_preprocessing(
            mesh._internal_mesh, preprocessing._internal_preprocessing
        )
        return Mesh(processed_mesh)
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during preprocessing application: {e}")
