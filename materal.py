'''
Author: DyllanElliia
Date: 2023-09-11 15:02:21
LastEditors: DyllanElliia
LastEditTime: 2023-09-14 14:54:08
Description:
'''
import taichi as ti
from taichi.math import *
from .sampler import hemispheric_sampling, roughness_sampling, fresnel_schlick ,hemispheric_sampling_roughness, pow5
from .ray import Ray

ENV_IOR = 1.000277


@ti.dataclass
class Material:
    albedo: vec3
    emission: vec3
    roughness: float
    metallic: float
    transmission: float
    ior: float

    @ti.func
    def BxDF(self, ray: Ray, normal: vec3, position: vec3) -> Ray:
        albedo = self.albedo
        roughness = self.roughness
        metallic = self.metallic
        transmission = self.transmission
        ior = self.ior

        outer = dot(ray.direction, normal) < 0
        normal *= 1 if outer else -1

        N = normal
        I = ray.direction
        V = -ray.direction
        NoI = dot(N, I)
        NoV = dot(N, V)


        if ti.random() < transmission:
            eta = ENV_IOR / ior
            outer = sign(NoV)  # 大于零就是穿入物体，小于零是穿出物体
            eta = pow(eta, outer)
            N *= outer  # 如果是穿出物体表面，就更改法线方向

            NoI = -NoV
            k = 1.0 - eta * eta * (1.0 - NoI * NoI)

            F0 = (eta - 1) / (eta + 1)
            F0 *= F0
            F = fresnel_schlick(NoI, F0, roughness)
            N = hemispheric_sampling_roughness(N, roughness)  # 根据粗糙度抖动法线方向

            if ti.random() < F + metallic and outer > 0 or k < 0:
                ray.direction = reflect(I, N)
            else:

                ray.direction = eta * I - (eta * NoI + sqrt(k)) * N
        else:
            eta = ENV_IOR / ior if outer else ior / ENV_IOR
            F0 = (eta - 1) / (eta + 1)
            F0 *= F0
            F = fresnel_schlick(NoI, F0, roughness)
            if ti.random() < F + metallic:
                N = hemispheric_sampling_roughness(N, roughness)
                ray.direction = reflect(I, N)

            else:
                ray.direction = hemispheric_sampling(N)

        ray.color *= albedo
        ray.origin = position
        ray.direction = normalize(ray.direction)
        return ray
