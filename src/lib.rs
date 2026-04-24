extern crate log;
extern crate mesh_projector_core;
extern crate mesh_projector_cpu;
extern crate mesh_projector_wgpu;
extern crate numpy;
extern crate pyo3;
extern crate rayon;

mod py_wrappers;

use mesh_projector_core::config::{PointCloudConfig, ProjectionConfig};
use mesh_projector_core::logging;
use mesh_projector_core::mesh_io;
use mesh_projector_core::preprocessing;
use mesh_projector_core::types::image_batch::ImageBatch;
use mesh_projector_core::types::mesh::Scene;
use mesh_projector_wgpu::WgpuOwnedMappedPointCloudReadback;
use mesh_projector_wgpu::{
    PlaneView, WgpuOutputSelection, WgpuOwnedMappedImageReadback, WgpuProjector,
};
use ndarray::{Array2, Array4, ArrayView2, ArrayView3, ArrayView4, ShapeBuilder};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4, PyArrayMethods};
use py_wrappers::{MeshPreprocessingWrapper, MeshWrapper};
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

#[pyclass(unsendable)]
struct PyProjectionReadbackOwner {
    readback: WgpuOwnedMappedImageReadback,
}

#[pyclass(unsendable)]
struct PyPointCloudReadbackOwner {
    readback: WgpuOwnedMappedPointCloudReadback,
}

#[derive(Clone, Copy)]
struct BorrowedPlaneMeta {
    ptr: *const u8,
    stride_bytes: usize,
}

fn plane_meta(plane: PlaneView<'_>, expected_bytes_per_view: usize) -> PyResult<BorrowedPlaneMeta> {
    if plane.bytes_per_view < expected_bytes_per_view {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "wgpu readback plane is too small: expected at least {} bytes per view, got {}",
            expected_bytes_per_view, plane.bytes_per_view
        )));
    }
    if plane.stride_bytes < expected_bytes_per_view {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "wgpu readback plane stride is too small: expected at least {} bytes, got {}",
            expected_bytes_per_view, plane.stride_bytes
        )));
    }
    Ok(BorrowedPlaneMeta {
        ptr: plane.bytes.as_ptr(),
        stride_bytes: plane.stride_bytes,
    })
}

fn strided_plane_meta(
    plane: PlaneView<'_>,
    expected_bytes: usize,
    expected_stride_bytes: usize,
) -> PyResult<BorrowedPlaneMeta> {
    if plane.bytes_per_view < expected_bytes {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "wgpu readback plane is too small: expected at least {} bytes, got {}",
            expected_bytes, plane.bytes_per_view
        )));
    }
    if plane.stride_bytes < expected_stride_bytes {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "wgpu readback plane stride is too small: expected at least {} bytes, got {}",
            expected_stride_bytes, plane.stride_bytes
        )));
    }
    Ok(BorrowedPlaneMeta {
        ptr: plane.bytes.as_ptr(),
        stride_bytes: plane.stride_bytes,
    })
}

fn make_array4_readonly<'py>(
    py: Python<'py>,
    meta: BorrowedPlaneMeta,
    owner: &Py<PyProjectionReadbackOwner>,
    views: usize,
    resolution: usize,
) -> Py<PyArray4<u8>> {
    let shape =
        (views, resolution, resolution, 4).strides((meta.stride_bytes, resolution * 4, 4, 1));
    let view = unsafe { ArrayView4::from_shape_ptr(shape, meta.ptr) };
    let array = unsafe {
        PyArray4::borrow_from_array(&view, owner.clone_ref(py).into_bound(py).into_any())
    };
    array.readwrite().make_nonwriteable();
    array.into()
}

fn make_array3_readonly<'py>(
    py: Python<'py>,
    meta: BorrowedPlaneMeta,
    owner: &Py<PyProjectionReadbackOwner>,
    views: usize,
    resolution: usize,
) -> Py<PyArray3<u8>> {
    let shape = (views, resolution, resolution).strides((meta.stride_bytes, resolution, 1));
    let view = unsafe { ArrayView3::from_shape_ptr(shape, meta.ptr) };
    let array = unsafe {
        PyArray3::borrow_from_array(&view, owner.clone_ref(py).into_bound(py).into_any())
    };
    array.readwrite().make_nonwriteable();
    array.into()
}

fn make_array2_f32_readonly<'py>(
    py: Python<'py>,
    meta: BorrowedPlaneMeta,
    owner: &Py<PyPointCloudReadbackOwner>,
    rows: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let element_size = std::mem::size_of::<f32>();
    if meta.stride_bytes % element_size != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "wgpu readback position stride is not f32-aligned: {} bytes",
            meta.stride_bytes
        )));
    }
    let shape = (rows, 3).strides((meta.stride_bytes / element_size, 1));
    let view = unsafe { ArrayView2::from_shape_ptr(shape, meta.ptr.cast::<f32>()) };
    let array = unsafe {
        PyArray2::borrow_from_array(&view, owner.clone_ref(py).into_bound(py).into_any())
    };
    array.readwrite().make_nonwriteable();
    Ok(array.into())
}

fn make_array2_u8_readonly<'py>(
    py: Python<'py>,
    meta: BorrowedPlaneMeta,
    owner: &Py<PyPointCloudReadbackOwner>,
    rows: usize,
) -> Py<PyArray2<u8>> {
    let shape = (rows, 3).strides((meta.stride_bytes, 1));
    let view = unsafe { ArrayView2::from_shape_ptr(shape, meta.ptr) };
    let array = unsafe {
        PyArray2::borrow_from_array(&view, owner.clone_ref(py).into_bound(py).into_any())
    };
    array.readwrite().make_nonwriteable();
    array.into()
}

fn wgpu_readback_into_numpy<'py>(
    py: Python<'py>,
    readback: WgpuOwnedMappedImageReadback,
) -> PyResult<(
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
)> {
    let views = readback.num_views;
    let resolution = readback.resolution;
    let pixels_per_view = resolution * resolution;
    let rgba_bytes_per_view = pixels_per_view * 4;

    let owner = Py::new(py, PyProjectionReadbackOwner { readback })?;
    let (color, depth, normal, occupancy) = {
        let borrowed = owner.bind(py).borrow();
        let readback = &borrowed.readback;
        (
            plane_meta(
                readback.color().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "wgpu readback did not contain a color plane",
                    )
                })?,
                rgba_bytes_per_view,
            )?,
            plane_meta(
                readback.depth().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "wgpu readback did not contain a depth plane",
                    )
                })?,
                pixels_per_view,
            )?,
            plane_meta(
                readback.normal().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "wgpu readback did not contain a normal plane",
                    )
                })?,
                rgba_bytes_per_view,
            )?,
            plane_meta(
                readback.occupancy().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "wgpu readback did not contain an occupancy plane",
                    )
                })?,
                pixels_per_view,
            )?,
        )
    };

    Ok((
        make_array4_readonly(py, color, &owner, views, resolution),
        make_array3_readonly(py, depth, &owner, views, resolution),
        make_array4_readonly(py, normal, &owner, views, resolution),
        make_array3_readonly(py, occupancy, &owner, views, resolution),
    ))
}

fn wgpu_point_cloud_readback_into_numpy<'py>(
    py: Python<'py>,
    readback: WgpuOwnedMappedPointCloudReadback,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<u8>>)> {
    let num_points = readback.num_points;
    if num_points == 0 {
        let positions = Array2::<f32>::zeros((0, 3)).into_pyarray(py);
        positions.readwrite().make_nonwriteable();
        let colors = Array2::<u8>::zeros((0, 3)).into_pyarray(py);
        colors.readwrite().make_nonwriteable();
        return Ok((positions.into(), colors.into()));
    }

    let positions_bytes = num_points * std::mem::size_of::<[f32; 4]>();
    let colors_bytes = num_points * std::mem::size_of::<u32>();
    let owner = Py::new(py, PyPointCloudReadbackOwner { readback })?;
    let (positions, colors) = {
        let borrowed = owner.bind(py).borrow();
        let readback = &borrowed.readback;
        (
            strided_plane_meta(
                readback.positions(),
                positions_bytes,
                std::mem::size_of::<[f32; 4]>(),
            )?,
            strided_plane_meta(readback.colors(), colors_bytes, std::mem::size_of::<u32>())?,
        )
    };

    Ok((
        make_array2_f32_readonly(py, positions, &owner, num_points)?,
        make_array2_u8_readonly(py, colors, &owner, num_points),
    ))
}

fn make_owned_array4_readonly<'py>(py: Python<'py>, array: Array4<u8>) -> Py<PyArray4<u8>> {
    let array = array.into_pyarray(py);
    array.readwrite().make_nonwriteable();
    array.into()
}

fn make_owned_array3_readonly<'py>(
    py: Python<'py>,
    array: ndarray::Array3<u8>,
) -> Py<PyArray3<u8>> {
    let array = array.into_pyarray(py);
    array.readwrite().make_nonwriteable();
    array.into()
}

fn image_batch_into_rgba_numpy<'py>(
    py: Python<'py>,
    image_batch: ImageBatch,
) -> (
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
    Py<PyArray4<u8>>,
    Py<PyArray3<u8>>,
) {
    (
        make_owned_array4_readonly(py, image_batch.images),
        make_owned_array3_readonly(py, image_batch.depth_maps),
        make_owned_array4_readonly(py, image_batch.normal_maps),
        make_owned_array3_readonly(py, image_batch.occupancy_maps),
    )
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
    let preprocessing =
        preprocessing::build_preprocessing_from_scene(&mesh, render_config.resolution);
    log::info!("Scene loaded successfully");
    Ok(MeshWrapper {
        mesh,
        preprocessing,
        resolution: render_config.resolution,
        use_gpu_renderer: render_config.use_gpu_renderer,
        is_preprocessed: false,
    })
}

#[pyfunction]
fn build_preprocessing(mesh_wrapper: &MeshWrapper, resolution: u16) -> MeshPreprocessingWrapper {
    MeshPreprocessingWrapper {
        preprocessing: preprocessing::build_preprocessing_from_scene(
            &mesh_wrapper.mesh,
            resolution,
        ),
    }
}

#[pyfunction]
fn apply_preprocessing(
    mesh_wrapper: &MeshWrapper,
    preprocessing: &MeshPreprocessingWrapper,
) -> MeshWrapper {
    MeshWrapper {
        mesh: preprocessing::preprocessed_scene(&mesh_wrapper.mesh, &preprocessing.preprocessing),
        preprocessing: preprocessing.preprocessing,
        resolution: mesh_wrapper.resolution,
        use_gpu_renderer: mesh_wrapper.use_gpu_renderer,
        is_preprocessed: true,
    }
}

fn scene_for_projection<'a>(
    mesh_wrapper: &'a MeshWrapper,
    preprocessing: Option<&MeshPreprocessingWrapper>,
) -> Result<std::borrow::Cow<'a, Scene>, PyErr> {
    if let Some(preprocessing) = preprocessing {
        if mesh_wrapper.is_preprocessed {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Cannot apply preprocessing at projection time to an already preprocessed mesh. Use the raw mesh returned by read_mesh, or call project_* without a preprocessing argument.",
            ));
        }
        return Ok(std::borrow::Cow::Owned(preprocessing::preprocessed_scene(
            &mesh_wrapper.mesh,
            &preprocessing.preprocessing,
        )));
    }

    if mesh_wrapper.is_preprocessed {
        Ok(std::borrow::Cow::Borrowed(&mesh_wrapper.mesh))
    } else {
        Ok(std::borrow::Cow::Owned(preprocessing::preprocessed_scene(
            &mesh_wrapper.mesh,
            &mesh_wrapper.preprocessing,
        )))
    }
}

#[pyfunction]
#[pyo3(signature = (mesh_wrapper, projection_config, view_angles, preprocessing=None))]
fn project_to_images<'py>(
    py: Python<'py>,
    mesh_wrapper: &MeshWrapper,
    projection_config: &Bound<'py, PyDict>,
    view_angles: &Bound<'py, pyo3::types::PyAny>,
    preprocessing: Option<&MeshPreprocessingWrapper>,
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
    let mesh = scene_for_projection(mesh_wrapper, preprocessing)?;
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
    if mesh_wrapper.use_gpu_renderer {
        let gpu_result = py.allow_threads(|| {
            shared_wgpu_projector().and_then(|projector| {
                projector
                    .project_to_owned_image_readback(
                        &mesh,
                        &settings,
                        &lighting_config,
                        WgpuOutputSelection::all(),
                    )
                    .map_err(|err| err.to_string())
            })
        });

        match gpu_result {
            Ok((readback, timings)) => {
                log::info!(
                    "Cube projections generation complete via zero-copy wgpu readback: total={:.3?}, readback={:.3?}, map_wait={:.3?}",
                    timings.total,
                    timings.readback,
                    timings.map_wait
                );
                return wgpu_readback_into_numpy(py, readback);
            }
            Err(err) => {
                log::warn!("wgpu image projection failed; falling back to CPU renderer: {err}");
            }
        }
    }

    let image_batch = py
        .allow_threads(|| mesh_projector_cpu::project_to_image(&mesh, &settings, &lighting_config));
    log::info!("Cube projections generation complete via CPU fallback!");
    Ok(image_batch_into_rgba_numpy(py, image_batch))
}

#[pyfunction]
#[pyo3(signature = (mesh_wrapper, projection_config, preprocessing=None))]
fn project_to_point_cloud<'py>(
    py: Python<'py>,
    mesh_wrapper: &MeshWrapper,
    projection_config: &Bound<'py, PyDict>,
    preprocessing: Option<&MeshPreprocessingWrapper>,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<u8>>)> {
    // Initialize logger if requested
    let proj_cfg = py_wrappers::parse_point_cloud_projection_config(projection_config);
    logging::atomic_init_logger(proj_cfg.verbose)?;
    log::info!("Entering Rust project_to_point_cloud backend");
    let mesh = scene_for_projection(mesh_wrapper, preprocessing)?;
    let resolution = mesh_wrapper.resolution;
    // Parse provided data from Python and convert to Rust types
    let lighting_config = proj_cfg.light_config;
    // Set up point cloud settings
    let settings = PointCloudConfig {
        resolution,
        point_cloud_density_factor: proj_cfg.density_factor,
    };
    if mesh_wrapper.use_gpu_renderer {
        let gpu_result = py.allow_threads(|| {
            shared_wgpu_projector().and_then(|projector| {
                projector
                    .project_to_owned_point_cloud_readback(&mesh, &settings, &lighting_config)
                    .map_err(|err| err.to_string())
            })
        });

        match gpu_result {
            Ok((readback, timings)) => {
                log::info!(
                    "Point-cloud generation complete via zero-copy wgpu readback: total={:.3?}, readback={:.3?}, map_wait={:.3?}",
                    timings.total,
                    timings.readback,
                    timings.map_wait
                );
                return wgpu_point_cloud_readback_into_numpy(py, readback);
            }
            Err(err) => {
                log::warn!(
                    "wgpu point cloud projection failed; falling back to CPU renderer: {err}"
                );
            }
        }
    }

    let (points, colors) = py.allow_threads(|| {
        mesh_projector_cpu::project_to_point_cloud(&mesh, &settings, &lighting_config)
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
    m.add_class::<MeshPreprocessingWrapper>()?;
    m.add_function(wrap_pyfunction!(read_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(build_preprocessing, m)?)?;
    m.add_function(wrap_pyfunction!(apply_preprocessing, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_images, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_point_cloud, m)?)?;
    Ok(())
}
