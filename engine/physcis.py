import numpy as np
import taichi as ti
import taichi_glsl as ts


@ti.func
def quaternion_multiply(a, b):
    w = a.w * b.w - ts.dot(a.xyz, b.xyz)
    v = a.w * b.xyz + b.w * a.xyz + ts.cross(a.xyz, b.xyz)
    return ti.Vector([v.x, v.y, v.z, w])


@ti.func
def quaternion_to_mat3(q):
    return ti.Matrix([
        1 - 2 * q.y * q.y - 2 * q.z * q.z, 2 * q.x * q.y -
        2 * q.w * q.z, 2 * q.x * q.z + 2 * q.w * q.y,
        2 * q.x * q.y + 2 * q.w * q.z, 1 - 2 * q.x * q.x -
        2 * q.z * q.z, 2 * q.y * q.z - 2 * q.w * q.x,
        2 * q.x * q.z - 2 * q.w * q.y, 2 * q.y * q.z + 2 *
        q.w * q.x, 1 - 2 * q.x * q.x - 2 * q.y * q.y
    ])
