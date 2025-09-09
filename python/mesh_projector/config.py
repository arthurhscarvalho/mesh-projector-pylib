from enum import StrEnum
from typing import Optional, Tuple, Union

from pydantic import BaseModel


class ShadingMode(StrEnum):
    VERTEX_COLOR = "vertex_color"
    TEXTURE_MAPPED = "texture_mapped"
    BLINN_PHONG = "blinn_phong"


class RenderConfig(BaseModel):
    shading_mode: Optional[ShadingMode] = None
    resolution: int = 4096
    flip_uv: bool = True
    ignore_opacity: bool = False
    verbose: bool = False


class LightMode(StrEnum):
    AMBIENT = "ambient"
    HEADLIGHT = "headlight"
    DIRECTIONAL = "directional"
    POINT = "point"


class LightBase(BaseModel):
    mode: LightMode
    global_ambient: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    global_diffuse: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    global_specular: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class AmbientLight(LightBase):
    mode: LightMode = LightMode.AMBIENT


class Headlight(LightBase):
    mode: LightMode = LightMode.HEADLIGHT


class DirectionalLight(LightBase):
    mode: LightMode = LightMode.DIRECTIONAL
    direction: Tuple[float, float, float]


class PointLight(LightBase):
    mode: LightMode = LightMode.POINT
    position: Tuple[float, float, float]


LightConfig = Union[AmbientLight, Headlight, DirectionalLight, PointLight]
