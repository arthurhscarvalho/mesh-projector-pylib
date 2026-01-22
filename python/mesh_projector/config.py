from enum import StrEnum
from typing import List, Literal, Tuple

from pydantic import BaseModel, confloat


class ShadingMode(StrEnum):
    VERTEX_COLOR = "vertex_color"
    TEXTURE_MAPPED = "texture_mapped"
    BLINN_PHONG = "blinn_phong"
    AUTO = "auto"


class RenderConfig(BaseModel):
    shading_mode: ShadingMode = ShadingMode.AUTO
    resolution: int = 4096
    flip_uv: bool = True
    ignore_opacity: bool = False
    verbose: bool = False


class CameraProjection(StrEnum):
    ORTHOGRAPHIC = "orthographic"
    PERSPECTIVE = "perspective"


class PerspectiveCamera(BaseModel):
    projection: Literal[CameraProjection.PERSPECTIVE] = CameraProjection.PERSPECTIVE
    vertical_fov_degrees: float


class OrthographicCamera(BaseModel):
    projection: Literal[CameraProjection.ORTHOGRAPHIC] = CameraProjection.ORTHOGRAPHIC


class MsaaSampleCount(StrEnum):
    X1 = "x1"
    X2 = "x2"
    X4 = "x4"
    X8 = "x8"


class MsaaConfig(BaseModel):
    sample_count: MsaaSampleCount = MsaaSampleCount.X1


class SsaoConfig(BaseModel):
    enabled: bool = False
    sample_radius: float = 0.75
    num_samples: int = 16
    bias: float = 0.025
    intensity: float = 0.5
    blur_radius: int = 4
    max_distance: float = 2.0


Channel = confloat(ge=0.0, le=1.0)
Alpha = confloat(ge=0.0, le=1.0)
RGB = Tuple[Channel, Channel, Channel]
RGBA = Tuple[Channel, Channel, Channel, Alpha]


class BackgroundConfig(BaseModel):
    color: RGB | RGBA = (1.0, 1.0, 1.0)


class LightMode(StrEnum):
    AMBIENT = "ambient"
    HEADLIGHT = "headlight"
    DIRECTIONAL = "directional"
    POINT = "point"


class LightBase(BaseModel):
    mode: str
    global_ambient: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    global_diffuse: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    global_specular: Tuple[float, float, float] = (0.5, 0.5, 0.5)


class AmbientLight(LightBase):
    mode: Literal[LightMode.AMBIENT] = LightMode.AMBIENT


class Headlight(LightBase):
    mode: Literal[LightMode.HEADLIGHT] = LightMode.HEADLIGHT


class DirectionalLight(LightBase):
    mode: Literal[LightMode.DIRECTIONAL] = LightMode.DIRECTIONAL
    direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)


class PointLights(LightBase):
    mode: Literal[LightMode.POINT] = LightMode.POINT
    positions: List[Tuple[float, float, float]] = [(-1.0, 1.0, 1.0), (1.0, -1.0, -1.0)]


LightConfig = AmbientLight | Headlight | DirectionalLight | PointLights
CameraConfig = OrthographicCamera | PerspectiveCamera


class ImageProjectionConfig(BaseModel):
    filtering_radius: int = 0
    light_config: LightConfig = Headlight()
    camera_config: CameraConfig = OrthographicCamera()
    msaa_config: MsaaConfig = MsaaConfig()
    ssao_config: SsaoConfig = SsaoConfig()
    background_config: BackgroundConfig = BackgroundConfig()
    verbose: bool = False


class PointCloudProjectionConfig(BaseModel):
    density_factor: float = 1.0
    light_config: LightConfig = PointLights()
    verbose: bool = False
