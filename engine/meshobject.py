import numpy as np
import igl
from util import *


class MeshObject:
    def __init__(self, name: str, path: str, position, rotation, polyscope) -> None:
        self.name = name
        v, f = igl.read_triangle_mesh(path)
        self.mesh = polyscope.register_surface_mesh(self.name, v, f)
        self.position = position
        self.orientation = euler_to_quaternion(rotation)
        self.scale = self.mesh.get_transform().diagonal()[:3]
        self.TRS = identity()
        self.compute_transform()

        self.mesh.set_transform(self.TRS)

    def compute_transform(self):
        self.TRS = TRS(self.position, self.orientation, self.scale)
