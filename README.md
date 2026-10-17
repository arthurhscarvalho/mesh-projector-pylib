# mesh-projector-pylib

## Purpose

Python bindings for the [mesh-projector-rs](https://github.com/arthurhscarvalho/mesh-projector-rs) Rust backend. The package loads OBJ meshes, applies reusable geometry preprocessing, renders image projections, and generates colored point clouds through surface sampling.

The PyPI distribution is named `mesh_projector`; its Python import is `mesh_projector`.

## Related projects

[mesh-projector-rs](https://github.com/arthurhscarvalho/mesh-projector-rs) provides the core mesh operations and CPU/GPU implementations. [MAPPE](https://github.com/arthurhscarvalho/mappe) uses this package to generate its image and point-cloud features.

## Requirements

- Python 3.11 or 3.12. Although [pyproject.toml](pyproject.toml) permits Python `>=3.10,<3.13`, the current configuration module imports `enum.StrEnum`, which requires Python 3.11 or later.
- NumPy, Pydantic, and the other declared runtime dependencies.
- For source builds: a current stable Rust toolchain, Cargo, a native linker/C compiler, Git, and Maturin.
- For GPU execution: a `wgpu`-compatible GPU adapter and drivers; Linux typically requires a working Vulkan driver stack.

The default renderer uses the CPU.

## Installation

### From PyPI

In a Python 3.11 or 3.12 environment:

```bash
python -m pip install mesh_projector
```

A compatible wheel does not require a local Rust toolchain. To build from source, use the sibling checkouts below.

### From source

```bash
git clone https://github.com/arthurhscarvalho/mesh-projector-rs.git
git clone https://github.com/arthurhscarvalho/mesh-projector-pylib.git
cd mesh-projector-pylib
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin>=1.13.1,<2.0"
python -m maturin develop --release
```

Keep `mesh-projector-rs/` and `mesh-projector-pylib/` as sibling directories. [Cargo.toml](Cargo.toml) references `../mesh-projector-rs/crates/core`, `crates/cpu`, and `crates/wgpu`. The activation command above is for a POSIX shell. Re-run `python -m maturin develop --release` after changing the binding or its backend.

## Usage

### Mesh preprocessing and projection

Run the following example from the repository root to use the bundled asset. For a PyPI-only installation, replace the OBJ path with your own file and preserve its material and texture paths.

```python
import math
import mesh_projector as mp

render_config = mp.RenderConfig(resolution=256, use_gpu_renderer=False)
mesh = mp.read_mesh("assets/just_a_girl_C0-L5_deq_tri.obj", render_config)
preprocessing = mp.build_preprocessing(mesh, render_config)
mesh = mp.apply_preprocessing(mesh, preprocessing)

view_angles = [
    (0.0, 0.0, 0.0),
    (0.0, math.pi, 0.0),
    (0.0, math.pi / 2.0, 0.0),
    (0.0, -math.pi / 2.0, 0.0),
    (math.pi / 2.0, 0.0, 0.0),
    (-math.pi / 2.0, 0.0, 0.0),
]
images, depths, normals, occupancies = mp.project_to_images(
    mesh, mp.ImageProjectionConfig(), view_angles
)
points, colors = mp.project_to_point_cloud(
    mesh, mp.PointCloudProjectionConfig(density_factor=0.01)
)

print(images.shape)  # (6, 256, 256, 4)
print(points.shape, colors.shape)  # Both (N, 3)
```

For paired quality assessment, build the preprocessing recipe from the reference mesh and apply that same recipe to both meshes. `apply_preprocessing` returns a new mesh. Alternatively, pass the recipe as the optional `preprocessing` argument to a projection function on a raw mesh.

| Output | Shape and dtype |
| --- | --- |
| Image projections | `(V, H, W, 4)`, `uint8` RGBA. |
| Depth maps | `(V, H, W)`, `uint8`. |
| Normal maps | `(V, H, W, 4)`, `uint8`; use `[..., :3]` for the three normal channels. |
| Occupancy maps | `(V, H, W)`, `uint8`; `255` denotes mesh coverage. |
| Sampled points | `(N, 3)`, `float32` XYZ coordinates. |
| Sampled colors | `(N, 3)`, `uint8` RGB values in `[0, 255]`. |

`V` is the number of supplied views; `H` and `W` equal `RenderConfig.resolution`. View angles are Euler rotations in radians. Treat image outputs as read-only. The point count depends on the mesh, resolution, and sampling density.

Set `use_gpu_renderer=True` when creating `RenderConfig` to request GPU rendering and sampling. If GPU initialization or projection fails, the Python extension logs a warning and falls back to the CPU implementation. Lighting, camera, antialiasing, ambient occlusion, and background settings are defined in [python/mesh_projector/config.py](python/mesh_projector/config.py).

### File export examples

Install the optional example dependencies, then run from the repository root:

```bash
python -m pip install open3d opencv-python
python examples/project_on_bounding_cube.py
python examples/project_to_point_cloud.py
```

[examples/project_on_bounding_cube.py](examples/project_on_bounding_cube.py) saves projection, depth, normal, and occupancy PNG files in `projections/`. [examples/project_to_point_cloud.py](examples/project_to_point_cloud.py) saves `point_cloud.ply`. These examples request the GPU renderer and use the bundled asset; edit their input paths and configurations as needed.

## Publications

The mesh processing and Triangular Area-Adaptive Point Sampling (TAPS) functionality is described as part of MAPPE:

1. Arthur Henrique Silva Carvalho and Pedro Garcia Freitas. “MAPPE: Mesh Assessment via Point Cloud and Projection Estimators.” *Signal Processing: Image Communication*, vol. 150, article 117722, 2027. [DOI: 10.1016/j.image.2026.117722](https://doi.org/10.1016/j.image.2026.117722). Available online 5 October 2026.

```bibtex
@article{carvalho2027mappe,
  author  = {Carvalho, Arthur Henrique Silva and Freitas, Pedro Garcia},
  title   = {{MAPPE}: Mesh Assessment via Point Cloud and Projection Estimators},
  journal = {Signal Processing: Image Communication},
  volume  = {150},
  pages   = {117722},
  year    = {2027},
  doi     = {10.1016/j.image.2026.117722}
}
```

## License

[MIT License](LICENSE).
