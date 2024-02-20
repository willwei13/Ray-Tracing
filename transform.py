'''
Author: DyllanElliia
Date: 2023-09-11 10:54:25
LastEditors: DyllanElliia
LastEditTime: 2023-09-14 14:47:58
Description: 
'''
import taichi as ti
from taichi.math import *


@ti.func
def EulerAngle2Mat3(a: vec3) -> mat3:
    s, c = sin(a), cos(a)
    print(a, s, c)

    return (
        mat3(vec3(c.z, s.z, 0), vec3(-s.z, c.z, 0), vec3(0, 0, 1))
        @ mat3(vec3(c.y, 0, -s.y), vec3(0, 1, 0), vec3(s.y, 0, c.y))
        @ mat3(vec3(1, 0, 0), vec3(0, c.x, s.x), vec3(0, -s.x, c.x))
    )


@ti.dataclass
class Transform:
    position: vec3
    rotation: vec3
    scale: vec3
    matrix: mat3
    i_matrix: mat3

    @ti.func
    def update_Mat(self):
        self.matrix = EulerAngle2Mat3(self.rotation)
        self.i_matrix = inverse(self.matrix)

    @ti.func
    def mul(self, p: vec3) -> vec3:
        p -= self.position  # Cannot squeeze the Euclidean space of distance field
        p = self.matrix @ p  # Otherwise the correct ray marching is not possible
        return p

    @ti.func
    def mul_dir(self, p: vec3) -> vec3:
        p = self.matrix @ p
        return p

    @ti.func
    def mul_dir_inv(self, p: vec3) -> vec3:
        p = self.i_matrix @ p
        return p
