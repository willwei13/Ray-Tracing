'''
Author: DyllanElliia
Date: 2023-09-11 15:02:21
LastEditors: DyllanElliia
LastEditTime: 2023-09-18 13:47:28
Description: 
'''
import taichi as ti
from taichi.math import *
from .transform import Transform
from .materal import Material
from .ray import Ray
from .sdfObj import MAX_DIS, MIN_DIS

import numpy as np
from plyfile import PlyData, PlyElement
import os

from .intersect.triangle import tra_intersects
from .intersect.sphere import sp_intersects, sp_cal_normal

MAX_DIS = 114514.0
MIN_DIS = 0.001919


def load_mesh(fn):
    print(f'loading {fn}')
    plydata = PlyData.read(fn)
    x = np.array(plydata['vertex']['x'])
    y = np.array(plydata['vertex']['y'])
    z = np.array(plydata['vertex']['z'])
    offset = (
        np.array([np.max(x) + np.min(x), np.max(y) + np.min(y), np.max(z) + np.min(z)])
        / 2.0
    )
    scale = np.max(
        [np.max(x) - np.min(x), np.max(y) - np.min(y), np.max(z) - np.min(z)]
    )
    elements = plydata['face']
    num_tris = len(elements['vertex_indices'])
    triangles = np.zeros((num_tris, 9), dtype=np.float32)

    for i, face in enumerate(elements['vertex_indices']):
        assert len(face) == 3
        for d in range(3):
            triangles[i, d * 3 + 0] = x[face[d]] * scale + offset[0]
            triangles[i, d * 3 + 1] = y[face[d]] * scale + offset[1]
            triangles[i, d * 3 + 2] = z[face[d]] * scale + offset[2]

    print('loaded')
    print(triangles)
    print(triangles.shape)
    return triangles


@ti.dataclass
class blackHoleObject:
    transform: Transform
    material: Material

    tra_end: int


@ti.data_oriented
class MeshObject:
    # def __init__(
    #     self, mesh_path: str, transform: Transform, material: Material
    # ) -> None:
    #     bh = blackHoleObject(transform, material, 0, 0)
    #     self.mesh_path = mesh_path

    #     self.obj_num = 0
    #     self.obj_bh = blackHoleObject.field()
    #     ti.root.dense(ti.i, self.obj_num).place(self.obj_bh)

    #     bh.tra_end = self.load_mesh(mesh_path)

    #     self.tr_num = 0
    #     self.host_trangles = None
    #     self.host_bh = []

    #     self.obj_bh[0] = bh
    #     self.updata_tf()

    # def load_mesh(self, path: str) -> float:
    #     mesh_trangles = load_mesh(path).reshape(-1, 3, 3).astype(np.float32)
    #     tr_num, _, _ = mesh_trangles.shape

    #     self.triangles = ti.field(vec3, shape=(tr_num, 3))
    #     self.triangles.from_numpy(mesh_trangles)
    #     self.tr_num = tr_num
    #     print('Loaded in Taichi, (', tr_num, ')')
    #     return tr_num

    def __init__(self) -> None:
        self.obj_num = 0

        self.tr_num = 0
        self.host_trangles = None
        self.host_bh = []

    def add_object(self, path: str, transform: Transform, material: Material) -> None:
        mesh_trangles = load_mesh(path).reshape(-1, 3, 3).astype(np.float32)
        if self.obj_num == 0:
            self.host_trangles = mesh_trangles
        else:
            self.host_trangles = np.append(self.host_trangles, mesh_trangles, axis=0)
        tr_num, _, _ = self.host_trangles.shape

        self.tr_num += tr_num
        tr_end = self.tr_num
        self.host_bh.append(blackHoleObject(transform, material, tr_end))
        self.obj_num += 1

    def load_2_device(self):
        self.triangles = ti.field(vec3, shape=(self.tr_num, 3))
        self.triangles.from_numpy(self.host_trangles)
        self.obj_bh = blackHoleObject.field()
        ti.root.dense(ti.i, self.obj_num).place(self.obj_bh)
        for i in range(self.obj_num):
            self.obj_bh[i] = self.host_bh[i]
        self.updata_tf()

    @ti.kernel
    def updata_tf(self):
        for i in ti.static(range(self.obj_num)):
            self.obj_bh[i].transform.update_Mat()

    @ti.func
    def get_triangle_i(self, i: int, scale: vec3) -> tuple[vec3, vec3, vec3]:
        return (
            self.triangles[i, 0] * scale,
            self.triangles[i, 1] * scale,
            self.triangles[i, 2] * scale,
        )

    @ti.func
    def tf_bh_i(self, i: int, org: vec3, dir: vec3) -> tuple[vec3, vec3, vec3]:
        return (
            self.obj_bh[i].transform.mul(org),
            self.obj_bh[i].transform.mul_dir(dir),
            self.obj_bh[i].transform.scale,
        )

    @ti.func
    def is_hit(self, r: Ray) -> tuple[Material, vec3, vec3, bool]:
        t = MAX_DIS
        obj_index = 0
        res_obj_index = 0
        res_index = 0
        is_hit = False
        # is_hit_i = False
        tra_norm = vec3(0)
        pos = vec3(0)

        r_org, r_dir, t_scale = self.tf_bh_i(0, r.origin, r.direction)
        for i in ti.static(range(self.tr_num)):
            # for i in ti.static(range(1)):

            if i >= self.obj_bh[obj_index].tra_end:
                obj_index += 1
                r_org, r_dir, t_scale = self.tf_bh_i(obj_index, r.origin, r.direction)
            # if ti.random() < 1e-8:
            #     print(r.origin, r_org, r.direction, r_dir)
            v0, v1, v2 = self.get_triangle_i(i, t_scale)
            # v0=vec3(0, 0, 0)
            # v1=vec3(1,0,0)
            # v2=vec3(0,1,0)
            is_hit_i, dis = tra_intersects(v0, v1, v2, r_org, r_dir)
            # if ti.random() < 1e-4 and abs(dis - MAX_DIS) > 1e-5:
            #     print(i, v0, r_org, dis)
            if is_hit_i and dis < t:
                # print("hit", i)
                res_index = i
                res_obj_index = obj_index
                t = dis
                is_hit = True
        if is_hit:
            t_scale = self.obj_bh[res_obj_index].transform.scale
            v0, v1, v2 = self.get_triangle_i(res_index, t_scale)
            # v0=vec3(0, 0, 0)
            # v1=vec3(1,0,0)
            # v2=vec3(0,1,0)
            tra_norm = self.calc_normal(v0, v1, v2)
            tra_norm = self.obj_bh[res_obj_index].transform.mul_dir_inv(tra_norm)
            pos = r.at(t)
        return self.obj_bh[res_obj_index].material, tra_norm, pos, is_hit

        # r_org, r_dir, t_scale = self.tf_bh_i(0, r.origin, r.direction)
        # is_hit_i, dis = sp_intersects(r_org, r_dir)
        # if is_hit_i and dis < t:
        #     t = dis
        #     is_hit = True
        # if is_hit_i:
        #     tra_norm = sp_cal_normal(r_org, r_dir, t)
        #     tra_norm=self.obj_bh[res_index].transform.mul_dir_inv(tra_norm)
        #     pos = r.at(t)
        # return self.obj_bh[0].material, tra_norm, pos, is_hit

    @ti.func
    def calc_normal(self, a: vec3, b: vec3, c: vec3) -> vec3:
        return normalize(cross(b - a, c - a))


# @ti.data_oriented
# class MeshWorld:
#   def __init__():
#     self.triangles=
