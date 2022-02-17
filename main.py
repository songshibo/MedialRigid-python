import polyscope as ps
import polyscope.imgui as psimgui
import os
from engine.simulator import Simulator
from util import *
from engine import *
import taichi as ti

ti.init(arch=ti.gpu, debug=True)

dt = 0.01
objects = []  # all active objects in scenes
selected = -1

root_path = os.getcwd()
ps_config = read_configs(
    os.path.join(root_path, "configs", "ps_config.ini"))
scene_config = read_configs(os.path.join(
    root_path, "scenes", "test_scene.ini"))

#######################
## polyscope configs ##
#######################
# options
ps.set_program_name(ps_config['options']['prog_name'])
ps.set_print_prefix(ps_config['options']['print_prefix'])
ps.set_max_fps(ps_config.getint('options', 'max_fps'))
ps.set_autoscale_structures(ps_config.getboolean('options', 'autoscale'))
# scene
ps.set_automatically_compute_scene_extents(
    ps_config.getboolean('scene', 'auto_compute_scene_extents'))
ps.set_length_scale(ps_config.getfloat('scene', 'length_scale'))
ps.set_bounding_box(read_np_array(
    ps_config['scene']['min'], np.float32), read_np_array(ps_config['scene']['max'], np.float32))
# Ground & Shadows
ps.set_ground_plane_mode(ps_config['scene']['ground_mode'])

########################
## Object preparation ##
########################
for obj_name in scene_config.sections():
    path = scene_config[obj_name]['path']
    position = read_np_array(scene_config[obj_name]['position'], np.float32)
    rotation = read_np_array(scene_config[obj_name]['rotation'], np.float32)
    obj = MeshObject(obj_name, path, position, rotation, len(objects), ps)
    obj.load_ma(scene_config[obj_name]['ma_path'])
    objects.append(obj)

sim = Simulator(dt, objects)


def callback():
    psimgui.PushItemWidth(150)

    if(psimgui.Button("Advance")):
        sim.update()
        # read back to polyscope
        result = sim.rigidbodies.to_numpy()
        for obj in objects:
            obj.update_transform(result)

    psimgui.PopItemWidth()

    # debug selected objects
    if ps.get_selection()[0] != '':
        psimgui.Separator()
        global selected
        name = ps.get_selection()[0]
        # update selection
        if selected == -1 or objects[selected].name != name:
            for idx, obj in enumerate(objects):
                if obj.name == name:
                    selected = idx
                    break
        # ui for debugging
        psimgui.PushItemWidth(200)
        psimgui.InputFloat3("Position", objects[selected].position)
        psimgui.InputFloat4("Orientation", objects[selected].orientation)
        psimgui.InputFloat3("Scale", objects[selected].scale)
        psimgui.PopItemWidth()


ps.init()
ps.set_user_callback(callback)
ps.show()
