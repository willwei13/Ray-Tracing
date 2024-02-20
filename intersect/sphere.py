'''
Author: DyllanElliia
Date: 2023-09-18 10:11:17
LastEditors: DyllanElliia
LastEditTime: 2023-09-18 10:37:47
Description: 
'''
import taichi as ti
from taichi.math import *
from ..ray import Ray
from ..sdfObj import MAX_DIS,MIN_DIS


@ti.func
def sp_intersects(rorg: vec3, rdir: vec3) -> tuple[bool, float]:
    is_hit = True
    t = MAX_DIS
    oc = rorg
    a = length(rdir)**2
    half_b = dot(oc, rdir)
    c = length(oc)**2 - 1
    discriminant = half_b **2 - a * c
    if discriminant < 0:
        is_hit = False
    else:
        sqrtd = sqrt(discriminant)
        root = (-half_b - sqrtd) / a
        if root<=0:
            root=(-half_b + sqrtd) / a
        t = root
        if t<MIN_DIS:
            is_hit=False
    return is_hit, t


@ti.func
def sp_cal_normal(rorg, rdir, t) -> vec3:
    return normalize(rorg + rdir * t)
