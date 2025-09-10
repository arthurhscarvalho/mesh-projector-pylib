use mesh_projector::renderer::config::{LightComponents, LightSource, LightingConfig};
use mesh_projector::types::color::Color;
use mesh_projector::types::coordinate::Coordinate;
use mesh_projector::types::mesh::{Scene, ShadingMode};
use mesh_projector::types::orientation::Orientation;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyDict;

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
    pub lighting_config: LightingConfig,
}
