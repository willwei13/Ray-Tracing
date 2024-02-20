'''
Author: DyllanElliia
Date: 2023-09-11 10:19:13
LastEditors: DyllanElliia
LastEditTime: 2023-09-11 10:47:58
Description: 
'''
import taichi as ti
from taichi.math import *

@ti.func
def random_in_unit_disk() -> vec2:
    x = ti.random()
    a = ti.random() * 2 * pi
    return sqrt(x) * vec2(sin(a), cos(a))