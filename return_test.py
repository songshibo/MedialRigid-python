import taichi as ti
from engine import *

ti.init(arch=ti.gpu)


def surface_distance(m1, m2):
    return np.linalg.norm(m1[:3] - m2[:3]) - (m1[3] + m2[3])


# first
m11 = np.array([0.0, 0.5, 0.5, 0.3]).astype(np.float32)
m12 = np.array([0.0, 1.0, 0.0, 0.45]).astype(np.float32)
# m13 = np.array([0.0, 0.0, 0.0, 1.0])
# second
m21 = np.array([0.5, 0.0, 0.0, 0.35]).astype(np.float32)
m22 = np.array([-0.5, 0.0, 0.0, 0.5]).astype(np.float32)
m23 = np.array([0.0, 0.0, -0.3, 0.2]).astype(np.float32)

u = UnitTest()

u.unit_sphere_slab(m11, m21, m22, m23)
tm = m21 * u.t21[None] + m22 * u.t22[None] + \
    m23 * (1.0 - u.t21[None] - u.t22[None])
print("surface_distance:{}".format(surface_distance(m11, tm)))

# u.unit_sphere_cone(m11, m21, m22)
# print(u.t21[None])
# tm = m21 * u.t21[None] + m22 * (1.0-u.t21[None])
# print(surface_distance(m11, tm))
# u.unit_sphere_cone(m11, m21, m23)
# tm = m21 * u.t21[None] + m23 * (1.0-u.t21[None])
# print(surface_distance(m11, tm))
# u.unit_sphere_cone(m11, m22, m23)
# tm = m22 * u.t21[None] + m23 * (1.0-u.t21[None])
# print(surface_distance(m11, tm))
