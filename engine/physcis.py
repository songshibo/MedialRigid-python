import numpy as np
import taichi as ti
import taichi_glsl as ts

g = ti.Vector([0, -9.8, 0])

ti.init(arch=ti.gpu)


@ti.kernel
def foo():
    num = 0
    for i, j, k, m in ti.ndrange((2, 4), (0, 2), (1, 4), (2, 5)):
        print("{},{},{},{}".format(i, j, k, m))
        num += 1
    print(num)


foo()


@ti.func
def quaternion_multiply(a, b):
    axyz = ts.vec3(a.x, a.y, a.z)
    bxyz = ts.vec3(b.x, b.y, b.z)
    w = a.w * b.w - ts.dot(axyz, bxyz)
    v = a.w * bxyz + b.w * axyz + ts.cross(axyz, bxyz)
    return ti.Vector([v.x, v.y, v.z, w])


@ti.func
def quaternion_to_mat3(q):
    return ti.Matrix([
        [1 - 2 * q.y * q.y - 2 * q.z * q.z, 2 * q.x * q.y -
            2 * q.w * q.z, 2 * q.x * q.z + 2 * q.w * q.y],
        [2 * q.x * q.y + 2 * q.w * q.z, 1 - 2 * q.x * q.x -
         2 * q.z * q.z, 2 * q.y * q.z - 2 * q.w * q.x],
        [2 * q.x * q.z - 2 * q.w * q.y, 2 * q.y * q.z + 2 *
         q.w * q.x, 1 - 2 * q.x * q.x - 2 * q.y * q.y]
    ])


@ti.func
def advance(rb, dt):
    rb.linear_m += rb.F * dt + g * rb.mass * dt
    rb.angular_m += rb.T * dt
    # linear velocity
    rb.v = rb.linear_m / rb.mass
    # angular velocity
    R = quaternion_to_mat3(rb.rot)
    rb.w = R @ rb.inv_I @ R.transpose() @ rb.angular_m
    qdot = quaternion_multiply(
        ti.Vector([rb.w.x, rb.w.y, rb.w.z, 0])*0.5, rb.rot)

    # update position & orientation
    rb.pos += rb.v * dt
    rb.rot += qdot * dt
    rb.rot = ts.normalize(rb.rot)

    # clear force & torque
    rb.F.fill(0.0)
    rb.T.fill(0.0)

    return rb
