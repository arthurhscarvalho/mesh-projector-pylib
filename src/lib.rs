extern crate log;
extern crate mesh_projector as mesh_projector_lib;
extern crate numpy;
extern crate pyo3;
extern crate rayon;

mod py_wrappers;

use mesh_projector_lib::logging;
use mesh_projector_lib::mesh_io;
use mesh_projector_lib::projector;
use mesh_projector_lib::projector::config::{PointCloudConfig, ProjectionConfig};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4};
use py_wrappers::MeshWrapper;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

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
    let image_batch =
        py.allow_threads(|| projector::project_to_image(&mesh, &settings, &lighting_config));
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
        projector::project_to_point_cloud_by_sampling(&mesh, &settings, &lighting_config)
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
