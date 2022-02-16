import taichi as ti
import taichi_glsl as ts
import numpy as np

from engine.physcis import quaternion_to_mat3

vec3 = ti.types.vector(3, ti.f32)
vec4 = ti.types.vector(4, ti.f32)
mat3 = ti.types.matrix(3, 3, ti.f32)


@ti.data_oriented
class Simulator:
    def __init__(self, dt, objects) -> None:
        self.dt = dt
        self.n = len(objects)
        rigidbodies_dict = self.prepare_data(
            objects)

        self.rigidbodies = ti.Struct.field({
            "pos": vec3,
            "rot": vec4,
            "mass": ti.f32,
            "inv_I": mat3,
            "linear_m": vec3,
            "angular_m": vec3,
            "v": vec3,
            "w": vec3,
            "F": vec3,
            "T": vec3,
        }, shape=self.n)

        self.rigidbodies.from_numpy(rigidbodies_dict)

        del rigidbodies_dict

    def prepare_data(self, objects):
        pos, rot, m, inv_I, linear_m, angular_m, v, omega, F, T = [
        ], [], [], [], [], [], [], [], [], []
        for obj in objects:
            pos.append(obj.position)
            rot.append(obj.orientation)
            m.append(1.0)
            inv_I.append(np.linalg.inv(obj.inertia))
            linear_m.append(np.array([0, 0, 0]))
            angular_m.append(np.array([0, 0, 0]))
            v.append(np.array([0, 0, 0]))
            omega.append(np.array([0, 0, 0]))
            F.append(np.array([0, 0, 0]))
            T.append(np.array([0, 0, 0]))
        return {
            "pos": np.array(pos),
            "rot": np.array(rot),
            "mass": np.array(m),
            "inv_I": np.array(inv_I),
            "linear_m": np.array(linear_m),
            "angular_m": np.array(angular_m),
            "v": np.array(v),
            "w": np.array(omega),
            "F": np.array(F),
            "T": np.array(T),
        }

    @ti.kernel
    def update(self):
        dt = self.dt
        g = ti.Vector([0, -9.8, 0])
        for i in range(self.n):
            rb = self.rigidbodies[i]

            rb.linear_m += rb.F * dt + g * rb.mass * dt
            rb.angular_m += rb.T * dt

            # linear velocity
            rb.v = rb.linear_m / rb.mass
            # angular velocity
            R = quaternion_to_mat3(ts.normalize(rb.rot))
            print(rb.rot)
            print(R)
            rb.pos += rb.v * dt

            self.rigidbodies[i] = rb

    @ti.kernel
    def debug(self):
        for i in range(self.n):
            print(self.rigidbodies[i].pos)
