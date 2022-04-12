import os

from engine.simulator import Simulator
from util import *
from engine import *
import taichi as ti
import polyscope as ps
import polyscope.imgui as psimgui

ti.init(arch=ti.cuda, kernel_profiler=True)


def load_ma(filepath):
    file = open(filepath, 'r')
    first_line = file.readline().rstrip()
    vcount, ecount, fcount = [int(x) for x in first_line.split()]
    assert vcount != 0, "No Medial Vertices!"
    # line number
    lineno = 1

    verts, primitives = [], []

    # read vertices
    i = 0
    while i < vcount:
        line = file.readline()
        # skip empty line or comment line
        if line.isspace() or line[0] == '#':
            lineno += 1
            continue
        v = line.split()
        # Exception
        assert v[0] == 'v', "vertex line: {} should start with \'v\'!".format(
            str(lineno))
        x, y, z, r = float(v[1]), float(v[2]), float(v[3]), float(v[4])
        verts.append(np.array([x, y, z, r]))
        lineno += 1
        i += 1
    # read edges
    i = 0
    while i < ecount:
        line = file.readline()
        if line.isspace() or line[0] == '#':
            lineno += 1
            continue
        e = line.split()
        # Exception
        assert e[0] == 'e', "Edge line: {} should start with \'e\'!".format(
            str(lineno))
        primitives.append(np.array([-1, int(e[1]), int(e[2])]))
        lineno += 1
        i += 1
    # read faces
    i = 0
    while i < fcount:
        line = file.readline()
        if line.isspace() or line[0] == '#':
            lineno += 1
            continue
        f = line.split()
        # Exception
        assert f[0] == 'f', "Face line: {} should start with \'f\'!".format(
            str(lineno))
        primitives.append(np.array([int(f[1]), int(f[2]), int(f[3])]))
        lineno += 1
        i += 1

    return np.array(verts, dtype=np.float32), np.array(primitives, dtype=np.int32)


d_verts, d_primtives = load_ma("./models/dinosaur.ma")
c_verts, c_primtives = load_ma("./models/dinosaur.ma")

len_dp = d_primtives.shape[0]
d_v = ti.Vector.field(n=4, dtype=ti.f32, shape=d_verts.shape[0])
d_p = ti.Vector.field(n=3, dtype=ti.int32, shape=d_primtives.shape[0])

len_cp = c_primtives.shape[0]
c_v = ti.Vector.field(n=4, dtype=ti.f32, shape=c_verts.shape[0])
c_p = ti.Vector.field(n=3, dtype=ti.int32, shape=c_primtives.shape[0])

d_v.from_numpy(d_verts)
d_p.from_numpy(d_primtives)
c_v.from_numpy(c_verts)
c_p.from_numpy(c_primtives)


@ti.kernel
def translate(x: ti.f32, y: ti.f32, z: ti.f32):
    for i in d_v:
        d_v[i][0] += x
        d_v[i][1] += y
        d_v[i][2] += z


@ti.kernel
def intersection_test():
    intersected = False
    for i, j in ti.ndrange(len_dp, len_cp):
        pi, pj = d_p[i], c_p[j]
        if pi[0] == -1:
            ij = detect_cone_cone(
                d_v[pi[1]], d_v[pi[2]], c_v[pj[1]], c_v[pj[2]])
            ti.atomic_or(intersected, ij)
        else:
            ij = detect_cone_slab(
                c_v[pj[1]], c_v[pj[2]], d_v[pi[0]], d_v[pi[1]], d_v[pi[2]])
            ti.atomic_or(intersected, ij)
    print(intersected)


@ti.kernel
def ccd_test():
    v11, v12, v13 = ti.Vector(
        [-3.0, 0.0, -3.0]), ti.Vector([-3.0, 0.0, -3.0]), ti.Vector([-3.0, 0.0, -3.0])
    v21, v22 = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
    toi = 1.0
    for i, j in ti.ndrange(len_dp, len_cp):
        pi, pj = d_p[i], c_p[j]
        if pi[0] == -1:
            t, _, _ = get_cone_cone_toi(d_v[pi[1]], d_v[pi[2]],
                                        c_v[pj[1]], c_v[pj[2]], v11, v12, v21, v22)
            ti.atomic_min(toi, t)
        else:
            t, _, _, _ = get_cone_slab_toi(
                c_v[pj[1]], c_v[pj[2]], d_v[pi[0]], d_v[pi[1]], d_v[pi[2]], v21, v22, v11, v12, v13)
            ti.atomic_min(toi, t)
    print(toi)


@ti.kernel
def ccd_test2():
    v11, v12, v13 = ti.Vector(
        [-3.0, 0.0, -3.0]), ti.Vector([-3.0, 0.0, -3.0]), ti.Vector([-3.0, 0.0, -3.0])
    v21, v22 = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
    toi = 1.0
    for k, l in ti.ndrange(37, 37):
        for i, j in ti.ndrange(len_dp, len_cp):
            pi, pj = d_p[i], c_p[j]
            if pi[0] == -1:
                t, _, _ = get_cone_cone_toi(d_v[pi[1]], d_v[pi[2]],
                                            c_v[pj[1]], c_v[pj[2]], v11, v12, v21, v22)
                ti.atomic_min(toi, t)
            else:
                t, _, _, _ = get_cone_slab_toi(
                    c_v[pj[1]], c_v[pj[2]], d_v[pi[0]], d_v[pi[1]], d_v[pi[2]], v21, v22, v11, v12, v13)
                ti.atomic_min(toi, t)


run = False


def callback():
    global run
    if psimgui.Button("Intersection Test"):
        intersection_test()

    if psimgui.Button("Performance Test"):
        ccd_test()
        # print(ti.profiler.get_kernel_profiler_total_time())
        ti.print_kernel_profile_info()
        ti.clear_kernel_profile_info()

    if psimgui.Button("Running"):
        run = not run

    if run:
        ccd_test()


translate(3, 0, 3)


ps.init()
ps.set_user_callback(callback)
ps.show()
