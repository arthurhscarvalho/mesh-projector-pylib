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
use mesh_projector_lib::types::orientation::Orientation;
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
    light_config: &Bound<'py, PyDict>,
) -> PyResult<MeshWrapper> {
    let render_config = py_wrappers::parse_render_config(render_config);
    let lighting_config = py_wrappers::parse_light_config(light_config);
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
        lighting_config,
    })
}

#[pyfunction]
fn project_to_image<'py>(
    py: Python<'py>,
    mesh_wrapper: &MeshWrapper,
    filtering_radius: u64,
    verbose: bool,
) -> PyResult<(
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
)> {
    // Initialize logger if requested
    logging::atomic_init_logger(verbose)?;
    log::info!("Entering Rust project_to_image_on_bounding_cube backend");
    let mesh = &mesh_wrapper.mesh;
    let resolution = mesh_wrapper.resolution;
    let lighting_config = &mesh_wrapper.lighting_config;
    // Set up view angles for cube projection
    const PI: f32 = std::f32::consts::PI;
    const VIEW_ANGLES: [[f32; 3]; 6] = [
        [0.0, 0.0, 0.0],       // Front: Cam at +Z, looking -Z. No rotation.
        [0.0, PI, 0.0],        // Back: Cam at -Z, looking +Z. Rotate 180 deg around Y.
        [0.0, PI / 2.0, 0.0],  // Right: Cam at +X, looking -X. Rotate 90 deg around Y.
        [0.0, -PI / 2.0, 0.0], // Left: Cam at -X, looking +X. Rotate -90 deg around Y.
        [PI / 2.0, 0.0, 0.0],  // Top: Cam at +Y, looking -Y. Rotate 90 deg around X.
        [-PI / 2.0, 0.0, 0.0], // Bottom: Cam at -Y, looking +Y. Rotate -90 deg around X.
    ];
    let view_angles = VIEW_ANGLES
        .iter()
        .map(|angle| Orientation::from(*angle))
        .collect();
    // Set up projection settings for cube projection
    let settings = ProjectionConfig {
        resolution,
        view_angles,
        filtering_window_radius: filtering_radius,
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
    point_cloud_density_factor: f32,
    verbose: bool,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<u8>>)> {
    // Initialize logger if requested
    logging::atomic_init_logger(verbose)?;
    log::info!("Entering Rust project_to_point_cloud backend");
    let mesh = &mesh_wrapper.mesh;
    let resolution = mesh_wrapper.resolution;
    let lighting_config = &mesh_wrapper.lighting_config;
    // Set up point cloud settings
    let settings = PointCloudConfig {
        resolution,
        point_cloud_density_factor,
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
    m.add_function(wrap_pyfunction!(project_to_image, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_point_cloud, m)?)?;
    Ok(())
}
