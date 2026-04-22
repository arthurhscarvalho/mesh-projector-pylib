extern crate log;
extern crate mesh_projector_core;
extern crate mesh_projector_cpu;
extern crate mesh_projector_wgpu;
extern crate numpy;
extern crate pyo3;
extern crate rayon;

mod py_wrappers;

use mesh_projector_core::config::{LightingConfig, PointCloudConfig, ProjectionConfig};
use mesh_projector_core::logging;
use mesh_projector_core::mesh_io;
use mesh_projector_core::types::image_batch::ImageBatch;
use mesh_projector_core::types::mesh::Scene;
use mesh_projector_wgpu::WgpuProjector;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4};
use py_wrappers::MeshWrapper;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use std::sync::OnceLock;

fn shared_wgpu_projector() -> Result<&'static WgpuProjector, String> {
    static WGPU_PROJECTOR: OnceLock<Result<WgpuProjector, String>> = OnceLock::new();
    WGPU_PROJECTOR
        .get_or_init(|| WgpuProjector::new_blocking().map_err(|err| err.to_string()))
        .as_ref()
        .map_err(|err| err.clone())
}

fn project_to_image_with_renderer(
    mesh: &Scene,
    settings: &ProjectionConfig,
    lighting_config: &LightingConfig,
    use_gpu_renderer: bool,
) -> ImageBatch {
    if use_gpu_renderer {
        match shared_wgpu_projector().and_then(|projector| {
            projector
                .project_to_image(mesh, settings, lighting_config)
                .map_err(|err| err.to_string())
        }) {
            Ok(image_batch) => return image_batch,
            Err(err) => {
                log::warn!("wgpu image projection failed; falling back to CPU renderer: {err}");
            }
        }
    }

    mesh_projector_cpu::project_to_image(mesh, settings, lighting_config)
}

fn project_to_point_cloud_with_renderer(
    mesh: &Scene,
    settings: &PointCloudConfig,
    lighting_config: &LightingConfig,
    use_gpu_renderer: bool,
) -> (Array2<f32>, Array2<u8>) {
    if use_gpu_renderer {
        match shared_wgpu_projector().and_then(|projector| {
            projector
                .project_to_point_cloud_by_sampling(mesh, settings, lighting_config)
                .map_err(|err| err.to_string())
        }) {
            Ok(point_cloud) => return point_cloud,
            Err(err) => {
                log::warn!(
                    "wgpu point cloud projection failed; falling back to CPU renderer: {err}"
                );
            }
        }
    }

    mesh_projector_cpu::project_to_point_cloud(mesh, settings, lighting_config)
}

#[pyfunction]
fn read_mesh<'py>(
    _py: Python<'py>,
    obj_path: &str,
    render_config: &Bound<'py, PyDict>,
) -> PyResult<MeshWrapper> {
    let render_config = py_wrappers::parse_render_config(render_config);
    // Initialize logger (if requested)
    logging::atomic_init_logger(render_config.verbose)?;
    log::info!("Entering Rust read_mesh backend");
    // Read the mesh
    let mesh = match mesh_io::read_mesh(
        obj_path,
        render_config.shading_mode,
        render_config.resolution,
        render_config.flip_uv,
        render_config.ignore_opacity,
    ) {
        Ok(mesh) => mesh,
        Err(e) => {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load mesh: {}",
                e
            )));
        }
    };
    log::info!("Scene loaded successfully");
    // Wrap the mesh in MeshWrapper
    Ok(MeshWrapper {
        mesh,
        resolution: render_config.resolution,
        use_gpu_renderer: render_config.use_gpu_renderer,
    })
}

#[pyfunction]
fn project_to_images<'py>(
    py: Python<'py>,
    mesh_wrapper: &MeshWrapper,
    projection_config: &Bound<'py, PyDict>,
    view_angles: &Bound<'py, pyo3::types::PyAny>,
) -> PyResult<(
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
)> {
    // Initialize logger if requested
    let proj_cfg = py_wrappers::parse_image_projection_config(projection_config);
    logging::atomic_init_logger(proj_cfg.verbose)?;
    log::info!("Entering Rust project_to_image_on_bounding_cube backend");
    let mesh = &mesh_wrapper.mesh;
    let resolution = mesh_wrapper.resolution;
    // Parse provided data from Python and convert to Rust types
    let view_angles = py_wrappers::parse_view_angles(view_angles);
    let lighting_config = proj_cfg.light_config;
    // Set up projection settings for cube projection
    let settings = ProjectionConfig {
        resolution,
        view_angles,
        filtering_window_radius: proj_cfg.filtering_radius,
        camera: proj_cfg.camera_config,
        msaa: proj_cfg.msaa_config,
        ssao: proj_cfg.ssao_config,
        background: proj_cfg.background_config,
    };
    // Call the backend to compute projections
    let image_batch = py.allow_threads(|| {
        project_to_image_with_renderer(
            mesh,
            &settings,
            &lighting_config,
            mesh_wrapper.use_gpu_renderer,
        )
    });
    let (projections, depth_maps, normal_maps, occlusion_maps) = (
        image_batch.images,
        image_batch.depth_maps,
        image_batch.normal_maps,
        image_batch.occupancy_maps,
    );
    log::info!("Cube projections generation complete!");
    // Convert the ndarray results directly to numpy arrays and return as a tuple
    Ok((
        projections.into_pyarray(py).into(),
        depth_maps.into_pyarray(py).into(),
        normal_maps.into_pyarray(py).into(),
        occlusion_maps.into_pyarray(py).into(),
    ))
}

#[pyfunction]
fn project_to_point_cloud<'py>(
    py: Python<'py>,
    mesh_wrapper: &MeshWrapper,
    projection_config: &Bound<'py, PyDict>,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<u8>>)> {
    // Initialize logger if requested
    let proj_cfg = py_wrappers::parse_point_cloud_projection_config(projection_config);
    logging::atomic_init_logger(proj_cfg.verbose)?;
    log::info!("Entering Rust project_to_point_cloud backend");
    let mesh = &mesh_wrapper.mesh;
    let resolution = mesh_wrapper.resolution;
    // Parse provided data from Python and convert to Rust types
    let lighting_config = proj_cfg.light_config;
    // Set up point cloud settings
    let settings = PointCloudConfig {
        resolution,
        point_cloud_density_factor: proj_cfg.density_factor,
    };
    // Call the backend to generate the point cloud
    let (points, colors) = py.allow_threads(|| {
        project_to_point_cloud_with_renderer(
            mesh,
            &settings,
            &lighting_config,
            mesh_wrapper.use_gpu_renderer,
        )
    });
    log::info!(
        "Back-end point cloud generation complete! Generated {} points",
        points.shape()[0]
    );
    // Convert the ndarray results directly to numpy arrays and return as a tuple
    Ok((
        points.into_pyarray(py).into(),
        colors.into_pyarray(py).into(),
    ))
}

#[pymodule]
fn mesh_projector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MeshWrapper>()?;
    m.add_function(wrap_pyfunction!(read_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_images, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_point_cloud, m)?)?;
    Ok(())
}
