'''
Author: DyllanElliia
Date: 2023-09-11 15:31:42
LastEditors: DyllanElliia
LastEditTime: 2023-11-06 11:30:34
Description: 
'''
import taichi as ti
from taichi.math import *
import rt

ti.init(arch=ti.gpu, default_ip=ti.i32, default_fp=ti.f32)
PI = 3.1415926
ires=5
image_resolution = (192 * ires, 108 * ires)

image_buffer = ti.Vector.field(4, float, image_resolution)
image_pixels = ti.Vector.field(3, float, image_resolution)

SCREEN_PIXEL_SIZE = 1.0 / vec2(image_resolution)
PIXEL_RADIUS = 0.5 * min(SCREEN_PIXEL_SIZE.x, SCREEN_PIXEL_SIZE.y)

MIN_DIS = 0.005
MAX_DIS = 2000.0
VISIBILITY = 0.000001

SAMPLE_PER_PIXEL = 1
MAX_RAYMARCH = 512
MAX_RAYTRACE = 512

SHAPE_SPHERE = 1
SHAPE_BOX = 2
SHAPE_CYLINDER = 3


aspect_ratio = image_resolution[0] / image_resolution[1]
light_quality = 128.0
camera_exposure = 1.0
camera_vfov = 30
camera_aperture = 0.01
camera_focus = 4
camera_gamma = 2.2

hdr_map = rt.Image('assets/Tokyo_BigSight_3k.hdr')
hdr_map.process(exposure=1.8, gamma=camera_gamma)

object_list = [
    rt.SDFObject(
        type=rt.TYPE_SPHERE,
        transform=rt.Transform(vec3(0, -100.501, 0), vec3(0), vec3(100)),
        material=rt.Material(vec3(1, 1, 1) * 0.6, vec3(1), 1, 1, 0, 1.635),
    ),
    rt.SDFObject(
        type=rt.TYPE_SPHERE,
        transform=rt.Transform(vec3(0, 0, 0), vec3(0), vec3(0.5)),
        material=rt.Material(vec3(1, 1, 1), vec3(0.1, 1, 0.1) * 10, 1, 0, 0, 1),
    ),
    rt.SDFObject(
        type=rt.TYPE_SPHERE,
        transform=rt.Transform(vec3(1, -0.2, 0), vec3(0), vec3(0.3)),
        material=rt.Material(vec3(0.2, 0.2, 1), vec3(1), 0.2, 1, 0, 1.100),
    ),
    rt.SDFObject(
        type=rt.TYPE_SPHERE,
        transform=rt.Transform(vec3(0.0, -0.2, 2), vec3(0), vec3(0.3)),
        material=rt.Material(vec3(1, 1, 1) * 0.9, vec3(1), 0, 0, 1, 1.5),
    ),
    rt.SDFObject(
        type=rt.TYPE_CYLINDER,
        transform=rt.Transform(vec3(-1.0, -0.2, 0), vec3(0), vec3(0.3)),
        material=rt.Material(vec3(1.0, 0.2, 0.2), vec3(1), 0, 0, 0, 1.460),
    ),
    rt.SDFObject(
        type=rt.TYPE_BOX,
        transform=rt.Transform(vec3(1.8, -0.4, 1.8), vec3(0), vec3(0.1)),
        material=rt.Material(vec3(1, 1, 1), vec3(1, 0.005, 0.005) * 30, 1, 0, 0, 1),
    ),
    rt.SDFObject(
        type=rt.TYPE_BOX,
        transform=rt.Transform(vec3(0, 0, 5), vec3(0), vec3(2, 1, 0.2)),
        material=rt.Material(vec3(1, 1, 0.2) * 0.9, vec3(1), 0, 1, 0, 0.470),
    ),
    rt.SDFObject(
        type=rt.TYPE_BOX,
        transform=rt.Transform(vec3(0, 0, -2), vec3(0), vec3(2, 1, 0.2)),
        material=rt.Material(vec3(1, 1, 1) * 0.9, vec3(1), 0, 1, 0, 2.950),
    ),
    # rt.SDFObject(
    #     type=rt.TYPE_BOX,
    #     transform=rt.Transform(
    #         vec3(1, -0.2, 1),
    #         vec3(-PI / 2, 0, PI / 2),
    #         vec3(0.6),
    #     ),
    #     material=rt.Material(vec3(1, 1, 1), vec3(1), 0, 0, 1, 1.5),
    # ),
    rt.SDFObject(
        type=rt.TYPE_BUNNY,
        transform=rt.Transform(
            vec3(1, -0.15, 1),
            vec3(-PI / 2, 0, PI / 2),
            vec3(0.5),
        ),
        material=rt.Material(vec3(1, 1, 0.2) * 0.9, vec3(1, 1, 1), 0.1, 0, 0.5, 1.5),
    ),
]

objects = rt.SDFObject.field()
objects_num = len(object_list)
ti.root.dense(ti.i, objects_num).place(objects)

for i in range(objects.shape[0]):
    objects[i] = object_list[i]


@ti.func
def update_transform():
    for i in objects:
        objects[i].transform.update_Mat()


@ti.kernel
def init_scene():
    update_transform()


@ti.func
def nearest_object(p: vec3) -> tuple[int, float]:
    index = 0
    min_dis = rt.MAX_DIS
    for i in ti.static(range(objects_num)):
        dis = abs(objects[i].signed_distance(p))
        if dis < min_dis:
            min_dis = dis
            index = i
    return index, min_dis


@ti.func
def raycast(ray: rt.Ray) -> tuple[rt.Material, vec3, vec3, bool]:
    t = rt.MIN_DIS
    index = 0
    position = vec3(0)
    hit = False
    for _ in range(MAX_RAYMARCH):
        position = ray.at(t)
        index, distance = nearest_object(position)
        # TODO: Ray takes a step forward
        t += distance
        # hit obj?

        hit = distance < 0.0000005
        # End condition
        if t > rt.MAX_DIS or hit:
            break
    obj = objects[index]
    return obj.material, obj.calc_normal(position), position, hit


@ti.func
def brightness(rgb: vec3) -> float:
    return dot(rgb, vec3(0.299, 0.587, 0.114))


@ti.func
def TBN(N: vec3) -> mat3:  # 用世界坐标下的法线计算 TBN 矩阵
    T = vec3(0)
    B = vec3(0)

    if N.z < -0.99999:
        T = vec3(0, -1, 0)
        B = vec3(-1, 0, 0)
    else:
        a = 1.0 / (1.0 + N.z)
        b = -N.x * N.y * a

        T = vec3(1.0 - N.x * N.x * a, b, -N.x)
        B = vec3(b, 1.0 - N.y * N.y * a, -N.y)

    return mat3(T, B, N)

@ti.func
def hemispheric_sampling(n: vec3) -> vec3:  # 以 n 为法线进行半球采样
    ra = ti.random() * 2 * pi
    rb = ti.random()

    rz = sqrt(rb)
    v = vec2(cos(ra), sin(ra))
    rxy = sqrt(1.0 - rb) * v

    return TBN(n) @ vec3(rxy, rz)


@ti.func
def raytrace(ray: rt.Ray) -> rt.Ray:
    for i in range(MAX_RAYTRACE):
        light_quality = 1 / 50
        inv_pdf = exp(float(i) * light_quality)

        roulette_prob = 1.0 - (1.0 / inv_pdf)
        visible = length(ray.color.rgb)
        # 如果光已经衰减到不可分辨程度，或者光线毙掉就不继续了
        if visible < 0.001 or ti.random() < roulette_prob:
            break
        # Raycast find nearest obj
        obj_material, obj_normal, position, hit = raycast(ray)
        # hit? if not, return sky box's intensity
        if not hit:
            ray.color *= hdr_map.texture(rt.sky_box_uv(ray))
            break
        ray = obj_material.BxDF(ray,obj_normal,position)
    return ray


@ti.kernel
def sample(camera_position: vec3, camera_lookat: vec3, camera_up: vec3):
    camera = rt.Camera()
    camera.lookfrom = camera_position
    camera.lookat = camera_lookat
    camera.vup = camera_up
    camera.aspect = aspect_ratio
    camera.vfov = camera_vfov
    camera.aperture = camera_aperture
    camera.focus = camera_focus

    for i, j in image_pixels:
        coord = vec2(i, j) + vec2(ti.random(), ti.random())
        uv = coord * SCREEN_PIXEL_SIZE

        ray = raytrace(camera.get_ray(uv, vec3(1)))
        image_buffer[i, j] += vec4(ray.color, 1.0)


@ti.kernel
def refresh():
    image_buffer.fill(vec4(0))


@ti.kernel
def render():
    for i, j in image_pixels:
        buffer = image_buffer[i, j]

        color = buffer.rgb / buffer.a
        color *= camera_exposure
        color = rt.ACESFitted(color)
        color = pow(color, vec3(1.0 / camera_gamma))

        image_pixels[i, j] = clamp(color, 0, 1)


window = ti.ui.Window("Taichi Renderer", image_resolution)
canvas = window.get_canvas()
camera = ti.ui.Camera()
camera.position(0, -0.2, 4)

init_scene()
frame = 0
while window.running:
    camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.LMB)
    moving = any(
        [window.is_pressed(key) for key in ('w', 'a', 's', 'd', 'q', 'e', 'LMB', ' ')]
    )
    if moving:
        refresh()

    for i in range(SAMPLE_PER_PIXEL):
        sample(camera.curr_position, camera.curr_lookat, camera.curr_up)
        print('frame:', frame, 'sample:', i + 1)
    frame += 1
    render()

    canvas.set_image(image_pixels)
    window.show()
