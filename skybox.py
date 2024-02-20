'''
Author: DyllanElliia
Date: 2023-09-11 10:19:06
LastEditors: DyllanElliia
LastEditTime: 2023-09-11 11:39:03
Description: 
'''
import taichi as ti
from taichi.math import *
from .ray import Ray
from .sampler import sample_spherical_map
from .image import Image

@ti.func
def sky_box_uv(ray: Ray) -> vec2:
    return sample_spherical_map(ray.direction)