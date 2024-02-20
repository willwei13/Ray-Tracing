import taichi as ti
from taichi.math import *


@ti.func
def sample_spherical_map(v: vec3) -> vec2:
    uv = vec2(atan2(v.z, v.x), asin(v.y))
    uv *= vec2(0.5 / pi, 1 / pi)
    uv += 0.5
    return uv


@ti.func
def fresnel_schlick(NoI: float, F0: float, roughness: float) -> float:
    return mix(mix(pow(abs(1.0 + NoI), 5.0), 1.0, F0), F0, roughness)



@ti.func
def hemispheric_sampling(normal: vec3) -> vec3:
    z = 2.0 * ti.random() - 1.0
    a = ti.random() * 2.0 * pi

    xy = sqrt(1.0 - z * z) * vec2(sin(a), cos(a))

    return normalize(normal + vec3(xy, z))

@ti.func
def hemispheric_sampling_roughness(normal: vec3, roughness: float) -> vec3:
    ra = ti.random() * 2 * pi
    rb = ti.random()

    shiny = pow5(roughness)  # 光感越大高光越锐利

    rz = sqrt((1.0 - rb) / (1.0 + (shiny - 1.0) * rb))  # 用粗糙度改变 Z 轴分布
    v = vec2(cos(ra), sin(ra))
    rxy = sqrt(abs(1 - rz * rz)) * v

    return normalize(normal + vec3(rxy, rz))


@ti.func
def pow5(x: float):
    t = x * x
    t *= t
    return t * x


@ti.func
def roughness_sampling(hemispheric_sample: vec3, normal: vec3, roughness: float) -> vec3:
    alpha = roughness * roughness
    return normalize(mix(normal, hemispheric_sample, alpha))