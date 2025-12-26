use mesh_projector_lib::projector::config::{
    BackgroundConfig, CameraConfig, MsaaConfig, MsaaSampleCount, PerspectiveCameraConfig,
    ProjectionType,
};
use mesh_projector_lib::renderer::config::{
    LightComponents, LightSource, LightingConfig, SsaoConfig,
};
use mesh_projector_lib::types::color::Color;
use mesh_projector_lib::types::coordinate::Coordinate;
use mesh_projector_lib::types::mesh::{Scene, ShadingMode};
use mesh_projector_lib::types::orientation::Orientation;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyDict;

pub struct ImageProjectionConfig {
    pub filtering_radius: u64,
    pub light_config: LightingConfig,
    pub camera_config: CameraConfig,
    pub msaa_config: MsaaConfig,
    pub ssao_config: SsaoConfig,
    pub background_config: BackgroundConfig,
    pub verbose: bool,
}

pub struct PointCloudProjectionConfig {
    pub density_factor: f32,
    pub light_config: LightingConfig,
    pub verbose: bool,
}

pub struct RenderConfig {
    pub shading_mode: Option<ShadingMode>,
    pub resolution: u16,
    pub flip_uv: bool,
    pub ignore_opacity: bool,
    pub verbose: bool,
}

pub fn parse_render_config<'py>(render_config: &Bound<'py, PyDict>) -> RenderConfig {
    // Getter for render config
    let rc_get =
        |key: &str| -> Option<Bound<'py, PyAny>> { render_config.get_item(key).ok().flatten() };
    let shading_mode_str: Option<String> =
        rc_get("shading_mode").and_then(|v| v.extract::<String>().ok());
    let resolution: u16 = rc_get("resolution")
        .and_then(|v| v.extract::<u16>().ok())
        .expect("Resolution must be provided");
    let flip_uv: bool = rc_get("flip_uv")
        .and_then(|v| v.extract::<bool>().ok())
        .expect("Flip UV must be provided");
    let ignore_opacity: bool = rc_get("ignore_opacity")
        .and_then(|v| v.extract::<bool>().ok())
        .expect("Ignore opacity must be provided");
    let verbose: bool = rc_get("verbose")
        .and_then(|v| v.extract::<bool>().ok())
        .expect("Verbose must be provided");
    let shading_mode: Option<ShadingMode> = shading_mode_str.as_deref().and_then(|s| match s {
        "vertex_color" => Some(ShadingMode::VertexColor),
        "texture_mapped" => Some(ShadingMode::TextureMapped),
        "blinn_phong" => Some(ShadingMode::BlinnPhong),
        _ => None,
    });
    RenderConfig {
        shading_mode,
        resolution,
        flip_uv,
        ignore_opacity,
        verbose,
    }
}

pub fn parse_light_config<'py>(light_config: &Bound<'py, PyDict>) -> LightingConfig {
    // Getter for light config
    let lc_get =
        |key: &str| -> Option<Bound<'py, PyAny>> { light_config.get_item(key).ok().flatten() };
    // Lighting config with defaults
    let global_ambient = lc_get("global_ambient")
        .and_then(|v| v.extract::<(f32, f32, f32)>().ok())
        .map(|(r, g, b)| Color::from_srgb(r, g, b))
        .expect("Global ambient color must be provided");
    let global_diffuse = lc_get("global_diffuse")
        .and_then(|v| v.extract::<(f32, f32, f32)>().ok())
        .map(|(r, g, b)| Color::from_srgb(r, g, b))
        .expect("Global diffuse color must be provided");
    let global_specular = lc_get("global_specular")
        .and_then(|v| v.extract::<(f32, f32, f32)>().ok())
        .map(|(r, g, b)| Color::from_srgb(r, g, b))
        .expect("Global specular color must be provided");
    // Determine light source
    let mode_str: String = lc_get("mode")
        .and_then(|v| v.extract::<String>().ok())
        .expect("Light mode must be provided");
    let source = match mode_str.as_str() {
        "ambient" => LightSource::Ambient,
        "headlight" => LightSource::Headlight,
        "directional" => {
            let dir: (f32, f32, f32) = lc_get("direction")
                .and_then(|v| v.extract::<(f32, f32, f32)>().ok())
                .expect("Directional light direction must be provided");
            LightSource::Directional(Coordinate::from(dir))
        }
        "point" => {
            // Expect a list of positions for multiple point lights
            let positions: Vec<(f32, f32, f32)> = lc_get("positions")
                .and_then(|v| v.extract::<Vec<(f32, f32, f32)>>().ok())
                .expect("Point light positions must be provided as a list of 3-tuples");
            let coords: Vec<Coordinate> = positions
                .into_iter()
                .map(|pos| Coordinate::from(pos))
                .collect();
            LightSource::Point(coords)
        }
        _ => LightSource::Headlight,
    };
    let lighting_config = LightingConfig {
        source,
        components: LightComponents {
            global_ambient,
            global_diffuse,
            global_specular,
        },
    };
    lighting_config
}

pub fn parse_camera_config<'py>(camera_config: Option<&Bound<'py, PyDict>>) -> CameraConfig {
    let mut cfg = CameraConfig::default();
    if let Some(camera_cfg) = camera_config {
        // Expect nested projection config (Pydantic model_dump), but fall back to flat keys.
        let projection_value = camera_cfg.get_item("projection").ok().flatten();
        // If nested, projection_value should be a dict with \"projection\" + params.
        if let Some(projection_value) = projection_value {
            if let Ok(projection_dict) = projection_value.downcast::<PyDict>() {
                let projection_str = projection_dict
                    .get_item("projection")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<String>().ok());
                cfg.projection = match projection_str.as_deref() {
                    Some("perspective") => {
                        let fov = projection_dict
                            .get_item("vertical_fov_degrees")
                            .ok()
                            .flatten()
                            .and_then(|v| v.extract::<f32>().ok())
                            .unwrap_or(70.0);
                        ProjectionType::Perspective(PerspectiveCameraConfig {
                            vertical_fov_degrees: fov,
                        })
                    }
                    _ => ProjectionType::Orthographic,
                };
            } else if let Ok(projection_str) = projection_value.extract::<String>() {
                // Backward-compatible: accept a flat projection string.
                cfg.projection = match projection_str.as_str() {
                    "perspective" => {
                        let fov = camera_cfg
                            .get_item("vertical_fov_degrees")
                            .ok()
                            .flatten()
                            .and_then(|v| v.extract::<f32>().ok())
                            .unwrap_or(70.0);
                        ProjectionType::Perspective(PerspectiveCameraConfig {
                            vertical_fov_degrees: fov,
                        })
                    }
                    _ => ProjectionType::Orthographic,
                };
            }
        }
    }
    cfg
}

pub fn parse_msaa_config<'py>(msaa_config: Option<&Bound<'py, PyDict>>) -> MsaaConfig {
    let mut cfg = MsaaConfig::default();
    if let Some(msaa_cfg) = msaa_config {
        let sample_str: Option<String> = msaa_cfg
            .get_item("sample_count")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok());
        cfg.sample_count = match sample_str.as_deref() {
            Some("x2") => MsaaSampleCount::X2,
            Some("x4") => MsaaSampleCount::X4,
            Some("x8") => MsaaSampleCount::X8,
            _ => MsaaSampleCount::X1,
        };
    }
    cfg
}

pub fn parse_ssao_config<'py>(ssao_config: Option<&Bound<'py, PyDict>>) -> SsaoConfig {
    let mut cfg = SsaoConfig::default();
    if let Some(ssao_cfg) = ssao_config {
        if let Some(enabled) = ssao_cfg
            .get_item("enabled")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<bool>().ok())
        {
            cfg.enabled = enabled;
        }
        if let Some(sample_radius) = ssao_cfg
            .get_item("sample_radius")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<f32>().ok())
        {
            cfg.sample_radius = sample_radius;
        }
        if let Some(num_samples) = ssao_cfg
            .get_item("num_samples")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<usize>().ok())
        {
            cfg.num_samples = num_samples;
        }
        if let Some(bias) = ssao_cfg
            .get_item("bias")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<f32>().ok())
        {
            cfg.bias = bias;
        }
        if let Some(intensity) = ssao_cfg
            .get_item("intensity")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<f32>().ok())
        {
            cfg.intensity = intensity;
        }
        if let Some(blur_radius) = ssao_cfg
            .get_item("blur_radius")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<usize>().ok())
        {
            cfg.blur_radius = blur_radius;
        }
        if let Some(max_distance) = ssao_cfg
            .get_item("max_distance")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<f32>().ok())
        {
            cfg.max_distance = max_distance;
        }
    }
    cfg
}

pub fn parse_background_config<'py>(
    background_config: Option<&Bound<'py, PyDict>>,
) -> BackgroundConfig {
    let mut cfg = BackgroundConfig::default();
    if let Some(bg_cfg) = background_config {
        if let Some(color) = bg_cfg
            .get_item("color")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<(f32, f32, f32)>().ok())
        {
            cfg.color = Color::from_srgb(color.0, color.1, color.2);
        }
        if let Some(alpha) = bg_cfg
            .get_item("alpha")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<u8>().ok())
        {
            cfg.alpha = alpha;
        }
    }
    cfg
}

pub fn parse_image_projection_config<'py>(
    projection_config: &Bound<'py, PyDict>,
) -> ImageProjectionConfig {
    let filtering_radius: u64 = projection_config
        .get_item("filtering_radius")
        .ok()
        .flatten()
        .and_then(|v| v.extract::<u64>().ok())
        .expect("filtering_radius must be provided");
    let verbose: bool = projection_config
        .get_item("verbose")
        .ok()
        .flatten()
        .and_then(|v| v.extract::<bool>().ok())
        .unwrap_or(false);
    let light_config_any = projection_config
        .get_item("light_config")
        .expect("light_config must be provided")
        .expect("light_config must be provided");
    let light_config_dict = light_config_any
        .downcast_into::<PyDict>()
        .expect("light_config must be provided as a dict");
    let camera_config_any = projection_config
        .get_item("camera_config")
        .expect("camera_config must be provided")
        .expect("camera_config must be provided");
    let camera_config_dict = camera_config_any
        .downcast_into::<PyDict>()
        .expect("camera_config must be provided as a dict");
    let msaa_config_any = projection_config
        .get_item("msaa_config")
        .expect("msaa_config must be provided")
        .expect("msaa_config must be provided");
    let msaa_config_dict = msaa_config_any
        .downcast_into::<PyDict>()
        .expect("msaa_config must be provided as a dict");
    let ssao_config_any = projection_config
        .get_item("ssao_config")
        .expect("ssao_config must be provided")
        .expect("ssao_config must be provided");
    let ssao_config_dict = ssao_config_any
        .downcast_into::<PyDict>()
        .expect("ssao_config must be provided as a dict");
    let background_config_any = projection_config
        .get_item("background_config")
        .expect("background_config must be provided")
        .expect("background_config must be provided");
    let background_config_dict = background_config_any
        .downcast_into::<PyDict>()
        .expect("background_config must be provided as a dict");
    let light_config = parse_light_config(&light_config_dict);
    let camera_config = parse_camera_config(Some(&camera_config_dict));
    let msaa_config = parse_msaa_config(Some(&msaa_config_dict));
    let ssao_config = parse_ssao_config(Some(&ssao_config_dict));
    let background_config = parse_background_config(Some(&background_config_dict));
    ImageProjectionConfig {
        filtering_radius,
        light_config,
        camera_config,
        msaa_config,
        ssao_config,
        background_config,
        verbose,
    }
}

pub fn parse_point_cloud_projection_config<'py>(
    projection_config: &Bound<'py, PyDict>,
) -> PointCloudProjectionConfig {
    let density_factor: f32 = projection_config
        .get_item("density_factor")
        .ok()
        .flatten()
        .and_then(|v| v.extract::<f32>().ok())
        .expect("density_factor must be provided");
    let verbose: bool = projection_config
        .get_item("verbose")
        .ok()
        .flatten()
        .and_then(|v| v.extract::<bool>().ok())
        .unwrap_or(false);
    let light_config_any = projection_config
        .get_item("light_config")
        .expect("light_config must be provided")
        .expect("light_config must be provided");
    let light_config_dict = light_config_any
        .downcast_into::<PyDict>()
        .expect("light_config must be provided as a dict");
    let light_config = parse_light_config(&light_config_dict);
    PointCloudProjectionConfig {
        density_factor,
        light_config,
        verbose,
    }
}

pub fn parse_view_angles<'py>(view_angles: &Bound<'py, PyAny>) -> Vec<Orientation> {
    // Expect a Python list/sequence of 3-tuples (roll, pitch, yaw) in radians
    let angles: Vec<(f32, f32, f32)> = view_angles
        .extract::<Vec<(f32, f32, f32)>>()
        .expect("view_angles must be a list of 3-tuples of floats");
    angles
        .into_iter()
        .map(|triple| Orientation::from(triple))
        .collect()
}

#[pyclass]
pub struct MeshWrapper {
    pub mesh: Scene,
    pub resolution: u16,
}
