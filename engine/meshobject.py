import igl
from util import *


def compute_mass_center(v):
    return np.mean(v, axis=0)


# Refer to http://number-none.com/blow/inertia/index.html
def compute_inertia_const_density(v, f, center):
    C_canonical = np.array([[1.0 / 60.0, 1.0 / 120.0, 1.0 / 120.0],
                            [1.0 / 120.0, 1.0 / 60.0, 1.0 / 120.0],
                            [1.0 / 120.0, 1.0 / 120.0, 1.0 / 60.0]])

    C_sum = np.zeros([3, 3])
    for tri in f:
        # compute Covariance of each tetrahedron
        v1 = v[tri[0], :]
        v2 = v[tri[1], :]
        v3 = v[tri[2], :]
        A = np.zeros([3, 3])
        A[:, 0] = v1 - center
        A[:, 1] = v2 - center
        A[:, 2] = v3 - center
        C = np.linalg.det(A) * A.dot(C_canonical).dot(A.T)
        C_sum = C_sum + C
    return np.trace(C_sum) * np.eye(3) - C_sum


class MeshObject:
    def __init__(self, name: str, path: str, position, rotation, index, polyscope) -> None:
        self.name = name
        self.index = index
        v, f = igl.read_triangle_mesh(path)
        self.mesh = polyscope.register_surface_mesh(self.name, v, f)
        self.position = position
        self.orientation = euler_to_quaternion(rotation)
        self.scale = self.mesh.get_transform().diagonal()[:3]
        self.TRS = identity()
        self.compute_transform()
        self.mesh.set_transform(self.TRS)

        self.mass_center = compute_mass_center(v)
        self.inertia = compute_inertia_const_density(v, f, self.mass_center)

    def compute_transform(self):
        self.TRS = TRS(self.position, self.orientation, self.scale)

    def update_transform(self, ti_result):
        self.position = ti_result['pos'][self.index]
        self.orientation = ti_result['rot'][self.index]
        self.compute_transform()
        self.mesh.set_transform(self.TRS)
