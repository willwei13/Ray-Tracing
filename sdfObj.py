'''
Author: DyllanElliia
Date: 2023-09-11 15:02:21
LastEditors: DyllanElliia
LastEditTime: 2023-11-06 11:15:29
Description: 
'''
import taichi as ti
from taichi.math import *
from .transform import Transform
from .materal import Material

MAX_DIS = 114514.0
MIN_DIS = 0.0001919

TYPE_SPHERE = 1


@ti.func
def sd_sphere(p: vec3, r: vec3) -> float:
    return length(p) - r.x


TYPE_BOX = 2


@ti.func
def sd_box(p: vec3, b: vec3) -> float:
    q = abs(p) - b
    return length(max(q, 0)) + min(max(q.x, max(q.y, q.z)), 0) - 0.03


TYPE_CYLINDER = 3


@ti.func
def sd_cylinder(p: vec3, rh: vec3) -> float:
    d = abs(vec2(length(p.xz), p.y)) - rh.xy
    return min(max(d.x, d.y), 0) + length(max(d, 0))


TYPE_BUNNY = 4


@ti.func
def sd_bunny(p: vec3, b: vec3) -> float:
    q = abs(p) - b
    return length(max(q, 0)) + min(max(q.x, max(q.y, q.z)), 0) - 0.03


#from .otherSdf.sdf_bunny import sd_bunny


@ti.dataclass
class SDFObject:
    type: int
    transform: Transform
    material: Material

    @ti.func
    def sdf_val(self, p: vec3, s: vec3) -> float:
        dis = MAX_DIS
        if self.type == TYPE_SPHERE:
            dis = sd_sphere(p, s)
        elif self.type == TYPE_BOX:
            dis = sd_box(p, s)
        elif self.type == TYPE_CYLINDER:
            dis = sd_cylinder(p, s)
        elif self.type == TYPE_BUNNY:
            dis = sd_bunny(p, s)
        return dis

    @ti.func
    def signed_distance(self, pos: vec3) -> float:
        return self.sdf_val(self.transform.mul(pos), self.transform.scale)

    @ti.func
    def calc_normal(self, p: vec3) -> vec3:
        e = vec2(1, -1) * 0.5773 * 0.005
        return normalize(
            e.xyy * self.signed_distance(p + e.xyy)
            + e.yyx * self.signed_distance(p + e.yyx)
            + e.yxy * self.signed_distance(p + e.yxy)
            + e.xxx * self.signed_distance(p + e.xxx)
        )

    @ti.func
    def pos2local(self, pos: vec3) -> float:
        pos_r = self.transform.mul(pos)
        return pos_r, self.transform.scale
