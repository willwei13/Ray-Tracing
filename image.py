'''
Author: DyllanElliia
Date: 2023-09-11 10:22:30
LastEditors: DyllanElliia
LastEditTime: 2023-09-11 11:30:49
Description: 
'''
import taichi as ti
from taichi.math import *


@ti.data_oriented
class Image:
    def __init__(self, path: str):
        img = ti.tools.imread(path).astype('float32')
        self.img = vec3.field(shape=img.shape[:2])
        self.img.from_numpy(img / 255)

    @ti.kernel
    def process(self, exposure: float, gamma: float):
        for i, j in self.img:
            color = self.img[i, j] * exposure
            color = pow(color, vec3(gamma))
            self.img[i, j] = color

    @ti.func
    def texture(self, uv: vec2) -> vec3:
        x = int(uv.x * self.img.shape[0])
        y = int(uv.y * self.img.shape[1])
        return self.img[x, y]