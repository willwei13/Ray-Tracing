'''
Author: DyllanElliia
Date: 2023-09-11 15:31:42
LastEditors: DyllanElliia
LastEditTime: 2023-11-06 11:32:39
Description: 
'''
import taichi as ti
from taichi.math import *
import rt

ti.init(arch=ti.gpu, default_ip=ti.i32, default_fp=ti.f32)
PI = 3.1415926
image_resolution = (192 * 5, 108 * 5)

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


tf = rt.Transform(vec3(0, 0, -2), vec3(0), vec3(1.5))
# object_mesh = rt.MeshObject(
#     mesh_path="./assets/fish_10.ply",
#     transform=tf,
#     material=rt.Material(vec3(0.2, 0.2, 1), vec3(1, 1, 1), 0.5, 0.8, 0.8, 1.5),
# )
object_mesh = rt.MeshObject()
object_mesh.add_object(
    "./assets/fish_10.ply",
    rt.Transform(vec3(0, 0, -2), vec3(0), vec3(1.5)),
    rt.Material(vec3(0.2, 0.2, 1), vec3(1, 1, 1), 0.5, 0.8, 0.8, 1.5),
)
object_mesh.add_object(
    "./assets/fish_10.ply",
    rt.Transform(vec3(1, 0, -2), vec3(0), vec3(1.5)),
    rt.Material(vec3(0.2, 0.2, 1), vec3(1, 1, 1), 0.5, 0.8, 0.8, 1.5),
)
object_mesh.load_2_device()


@ti.func
def update_transform():
    # print(tf.rotation)
    # print(object_mesh.transform.rotation)
    # object_mesh.transform.update_Mat()
    return


@ti.kernel
def init_scene():
    is_h, t = rt.tra_intersects(
        vec3(0, 0, 0),
        vec3(1, 0, 0),
        vec3(0, 1, 0),
        vec3(0.1, 0.1, -1),
        normalize(vec3(0.05, 0.08, 1)),
    )
    print(is_h, t)
    update_transform()


@ti.func
def brightness(rgb: vec3) -> float:
    return dot(rgb, vec3(0.299, 0.587, 0.114))




@ti.func
def raytrace(ray: rt.Ray) -> rt.Ray:
    # TODO: note, the api you need is object_mesh.is_hit(ray)
    for i in range(MAX_RAYTRACE):
        light_quality = 1/50
        inv_pdf = exp(float(i)*light_quality)
        roulette_prob = 1.0-(1.0/inv_pdf)
        visible = length(ray.color.rgb)

        if visible <0.001 or ti.random() < roulette_prob:
            break

        obj_material, obj_normal, position, hit = object_mesh.is_hit(ray)
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
