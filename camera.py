'''
Author: DyllanElliia
Date: 2023-09-11 10:18:12
LastEditors: DyllanElliia
LastEditTime: 2023-09-11 11:43:59
Description: 
'''
import taichi as ti
from taichi.math import *
from .ray import *
from .random import random_in_unit_disk

@ti.dataclass
class Camera:
    lookfrom: vec3
    lookat: vec3
    vup: vec3
    vfov: float
    aspect: float
    aperture: float
    focus: float
    
    @ti.func
    def get_ray(self, uv: vec2, color: vec3) -> Ray:
        theta = radians(self.vfov)
        half_height = tan(theta * 0.5)
        half_width = self.aspect * half_height

        z = normalize(self.lookfrom - self.lookat)
        x = normalize(cross(self.vup, z))
        y = cross(z, x)

        lens_radius = self.aperture * 0.5
        rud = lens_radius * random_in_unit_disk()
        offset = x * rud.x + y * rud.y
        
        hwfx = half_width  * self.focus * x
        hhfy = half_height * self.focus * y

        lower_left_corner = self.lookfrom - hwfx - hhfy - self.focus * z
        horizontal = 2.0 * hwfx
        vertical   = 2.0 * hhfy

        ro = self.lookfrom + offset
        po = lower_left_corner + uv.x * horizontal + uv.y * vertical
        rd = normalize(po - ro)

        return Ray(ro, rd, color)