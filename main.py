import polyscope as ps
import os
from util import *
from engine import *

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
objects = []  # all active objects in scenes
for obj_name in scene_config.sections():
    path = scene_config[obj_name]['path']
    position = read_np_array(scene_config[obj_name]['position'], np.float32)
    rotation = read_np_array(scene_config[obj_name]['rotation'], np.float32)
    objects.append(MeshObject(obj_name, path, position, rotation, ps))


ps.init()

ps.show()
