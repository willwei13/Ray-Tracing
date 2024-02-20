'''
Author: DyllanElliia
Date: 2023-09-11 20:37:09
LastEditors: DyllanElliia
LastEditTime: 2023-09-18 11:15:18
Description: 
'''
import taichi as ti
from taichi.math import *
from ..ray import Ray
from ..sdfObj import MAX_DIS

TRA_EPSILON = 1e-7


@ti.func
def tra_intersects(
    v0: vec3, v1: vec3, v2: vec3, rorg: vec3, rdir: vec3
) -> tuple[bool, float]:
    is_hit = True
    dis = MAX_DIS
    e1 = v1 - v0
    e2 = v2 - v0
    h = cross(rdir, e2)
    a = dot(e1, h)
    if a < -TRA_EPSILON or a > TRA_EPSILON:
        # print("in", a)
        f = 1 / a
        s = rorg - v0
        u = f * dot(s, h)
        if u < 0.0 or u > 1.0:
            is_hit = False
        else:
            q = cross(s, e1)
            v = f * dot(rdir, q)
            if v < 0 or u + v > 1.0:
                is_hit = False
            t = f * dot(e2, q)
            if t <= TRA_EPSILON:
                is_hit = False
            dis = t if is_hit else MAX_DIS
    else:
        is_hit = False
    return is_hit, dis


@ti.func
def tra_cal_normal(v0: vec3, v1: vec3, v2: vec3) -> vec3:
    return cross(v1 - v0, v2 - v0)
