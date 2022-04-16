from turtle import pos
from numpy import dtype
import taichi as ti
from taichi._lib import core as _ti_core
import igl

ti.init(arch=ti.gpu, dynamic_index=True)

gui_type = 'auto'

if _ti_core.GGUI_AVAILABLE:
    gui_type = 'ggui'
else:
    gui_type = 'cpu'

v, f = igl.read_triangle_mesh("models/armadillo.obj")
print("vertices: {}\nfaces: {}".format(len(v), len(f)))

x = ti.Vector.field(3, dtype=ti.f32, shape=len(v))
ox = ti.Vector.field(3, dtype=ti.f32, shape=len(v))
indices_lens = 3 * len(f)
indices = ti.field(dtype=ti.i32, shape=indices_lens)
ox.from_numpy(v)
indices.from_numpy(f.reshape(indices_lens))

position = ti.Vector.field(3, ti.f32, shape=1)


@ti.kernel
def update(trans: ti.template()):
    for u in x:
        x[u] = ox[u] + trans


if __name__ == '__main__':
    if gui_type == 'ggui':
        res = (800, 600)
        window = ti.ui.Window("Vulkan Backend", res, vsync=False)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.make_camera()
        camera.position(0.0, 0.0, 2.0)
        camera.lookat(0.0, 0.0, 0.0)
        camera.fov(60)

        def render():
            camera.track_user_inputs(window,
                                     movement_speed=0.01,
                                     hold_key=ti.ui.RMB)
            scene.set_camera(camera)

            scene.ambient_light((0.1, ) * 3)

            scene.point_light(pos=(0.5, 10.0, 0.5), color=(0.5, 0.5, 0.5))
            scene.point_light(pos=(10.0, 10.0, 10.0), color=(0.5, 0.5, 0.5))

            update(position[0])
            # scene.mesh(ox, indices, color=(0.73, 0.33, 0.23))
            scene.particles(x, radius=0.01)

            canvas.scene(scene)

        while window.running:
            window.GUI.begin("Simulator Parameters", 0.01, 0.01, 0.5, 0.4)
            position[0].x = window.GUI.slider_float("X", position[0].x, -1, 1)
            position[0].y = window.GUI.slider_float("Y", position[0].y, -1, 1)
            position[0].z = window.GUI.slider_float("Z", position[0].z, -1, 1)
            window.GUI.end()
            render()
            window.show()
    else:
        pass
