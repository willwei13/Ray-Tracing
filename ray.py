import taichi as ti
from taichi.math import *

@ti.dataclass
class Ray:
    origin: vec3
    direction: vec3
    color: vec3
    @ti.func
    def at(self, t: float) -> vec3:
      return self.origin + t * self.direction