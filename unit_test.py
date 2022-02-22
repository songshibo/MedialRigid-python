import polyscope as ps
import polyscope.imgui as psimgui
import os

from engine.simulator import Simulator
from util import *
from engine import *
import taichi as ti

ti.init(arch=ti.gpu, debug=True)


def surface_distance(m1, m2):
    return np.linalg.norm(m1[:3] - m2[:3]) - (m1[3] + m2[3])


def TS(mesh, _pos, _scale):
    T = translate_matrix(_pos)
    scale = np.ones(3) * _scale * 2.0
    S = scale_matrix(scale)
    mesh.set_transform(concatenate_matrices(T, S))


def register_sphere(name, m, v, f, color, enable=True):
    mesh = ps.register_surface_mesh(name, v, f)
    TS(mesh, m[:3], m[3])
    mesh.set_color(color)
    mesh.set_enabled(enable)
    mesh.set_smooth_shade(True)
    return mesh


def generate_edge(name, m1, m2):
    v = np.array([m1[:3], m2[:3]])
    e = np.array([[0, 1]])
    ps.register_curve_network(name, v, e)


def generate_triangle(name, m1, m2, m3):
    v = np.array([m1[:3], m2[:3], m3[:3]])
    e = np.array([[0, 1], [1, 2], [2, 0]])
    ps.remove_curve_network(name, error_if_absent=False)
    ps.register_curve_network(name, v, e)


vs, fs = igl.read_triangle_mesh("./models/unit_sphere.obj")

unit_test_options = ["Sphere-Cone", "Sphere-Slab", "Cone-Cone", "Cone-Slab"]
unit_test_selected = unit_test_options[0]


def update_curve_network():
    global unit_test_selected
    if unit_test_selected == "Sphere-Cone":
        ps.remove_curve_network("p1", error_if_absent=False)
        generate_edge("p2", m21, m22)
    elif unit_test_selected == "Sphere-Slab":
        ps.remove_curve_network("p1", error_if_absent=False)
        generate_triangle("p2", m21, m22, m23)
    elif unit_test_selected == "Cone-Cone":
        generate_edge("p1", m11, m12)
        generate_edge("p2", m21, m22)
    elif unit_test_selected == "Cone-Slab":
        generate_edge("p1", m11, m12)
        generate_triangle("p2", m21, m22, m23)


def decom_transform(m):
    pos = np.array(m[0:3, 3])
    m[0:3, 3] = np.zeros(3)
    M_without_rot = np.matmul(m, m.T)
    scale = np.sqrt(M_without_rot[0, 0]) * 0.5
    return np.array([pos[0], pos[1], pos[2], scale])


# first
m11 = np.array([0.0, 0.5, 0.5, 0.3])
m12 = np.array([0.0, 1.0, 0.0, 0.45])
# m13 = np.array([0.0, 0.0, 0.0, 1.0])
# second
m21 = np.array([0.5, 0.0, 0.0, 0.35])
m22 = np.array([-0.5, 0.0, 0.0, 0.5])
m23 = np.array([0.0, 0.0, -0.3, 0.2])

color1 = np.array([235.0, 64.0, 52.0, 255.0]) / 255.0
color2 = np.array([52.0, 177.0, 235.0, 255.0]) / 255.0
sp11 = register_sphere("m11", m11, vs, fs, color1, True)
sp12 = register_sphere("m12", m12, vs, fs, color1, False)
# sp13 = register_sphere("m13", m13, vs, fs, color1, False)
# first medial primitive lines

sp21 = register_sphere("m21", m21, vs, fs, color2, True)
sp22 = register_sphere("m22", m22, vs, fs, color2, True)
sp23 = register_sphere("m23", m23, vs, fs, color2, False)
# second medial primitive lines
generate_edge("p2", m21, m22)

u_ins = UnitTest()
steps = 5000


def callback():
    global m11, m12, m13, m21, m22, m23, unit_test_selected, steps

    if psimgui.Button("Update from Scene"):
        m11 = decom_transform(ps.get_surface_mesh(
            "m11").get_transform())
        m12 = decom_transform(ps.get_surface_mesh(
            "m12").get_transform())
        m21 = decom_transform(ps.get_surface_mesh(
            "m21").get_transform())
        m22 = decom_transform(ps.get_surface_mesh(
            "m22").get_transform())
        m23 = decom_transform(ps.get_surface_mesh(
            "m23").get_transform())
        update_curve_network()
    psimgui.SameLine()
    if psimgui.Button("Make Spheres Transparent"):
        sp11.set_transparency(0.5)
        sp12.set_transparency(0.5)
        sp21.set_transparency(0.5)
        sp22.set_transparency(0.5)
        sp23.set_transparency(0.5)
    psimgui.SameLine()
    if psimgui.Button("Make Spheres Solid"):
        sp11.set_transparency(1)
        sp12.set_transparency(1)
        sp21.set_transparency(1)
        sp22.set_transparency(1)
        sp23.set_transparency(1)

    psimgui.PushItemWidth(200)
    changed = psimgui.BeginCombo("Unit Test Type", unit_test_selected)
    if changed:
        for val in unit_test_options:
            _, selected = psimgui.Selectable(val, unit_test_selected == val)
            if selected:
                unit_test_selected = val
                update_curve_network()
        psimgui.EndCombo()
    psimgui.PopItemWidth()
    psimgui.Separator()

    psimgui.TextUnformatted("First Primitive")
    changed, m11 = psimgui.InputFloat4("Medial Sphere11", m11)
    if changed:
        TS(sp11, m11[:3], m11[3])
    if unit_test_selected == "Cone-Cone" or unit_test_selected == "Cone-Slab":
        sp12.set_enabled(True)
        changed, m12 = psimgui.InputFloat4("Medial Sphere12", m12)
        if changed:
            update_curve_network()
            TS(sp12, m12[:3], m12[3])
    else:
        sp12.set_enabled(False)
    # if unit_test_selected == "Slab-Slab":
    #     sp13.set_enabled(True)
    #     changed, m13 = psimgui.InputFloat4("Medial Sphere13", m13)
    #     if changed:
    #         TS(sp13, m13[:3], m13[3])
    # else:
    #     sp13.set_enabled(False)

    psimgui.Separator()
    psimgui.TextUnformatted("Second Primitive")
    changed, m21 = psimgui.InputFloat4("Medial Sphere21", m21)
    if changed:
        update_curve_network()
        TS(sp21, m21[:3], m21[3])
    changed, m22 = psimgui.InputFloat4("Medial Sphere22", m22)
    if changed:
        update_curve_network()
        TS(sp22, m22[:3], m22[3])
    if unit_test_selected == "Sphere-Slab" or unit_test_selected == "Cone-Slab":
        sp23.set_enabled(True)
        changed, m23 = psimgui.InputFloat4("Medial Sphere23", m23)
        if changed:
            update_curve_network()
            TS(sp23, m23[:3], m23[3])
    else:
        sp23.set_enabled(False)

    psimgui.Separator()
    m11, m12, m21, m22, m23 = np.array(m11, dtype=np.float32), np.array(
        m12, dtype=np.float32), np.array(m21, dtype=np.float32), np.array(m22, dtype=np.float32), np.array(m23, dtype=np.float32)
    if psimgui.Button("Compute Neareset"):
        if unit_test_selected == "Sphere-Cone":
            u_ins.unit_sphere_cone(m11, m21, m22)
            tm = m21 * u_ins.t21[None] + m22 * (1.0 - u_ins.t21[None])
            generate_edge("Nearest", m11, tm)
            print("[taichi] linear t:{}".format(u_ins.t21[None]))
            print("[taichi] minimum distance:{}".format(
                surface_distance(m11, tm)))
        elif unit_test_selected == "Sphere-Slab":
            u_ins.unit_sphere_slab(m11, m21, m22, m23)
            tm = m21 * u_ins.t21[None] + m22 * u_ins.t22[None] + \
                m23 * (1.0 - u_ins.t21[None] - u_ins.t22[None])
            generate_edge("Nearest", m11, tm)
            print("[taichi] barycentric t1,t2:{},{}".format(
                u_ins.t21[None], u_ins.t22[None]))
            print("[taichi] minimum distance:{}".format(
                surface_distance(m11, tm)))
        elif unit_test_selected == "Cone-Cone":
            u_ins.unit_cone_cone(m11, m12, m21, m22)
            tm1 = m11 * u_ins.t11[None] + m12 * (1.0 - u_ins.t11[None])
            tm2 = m21 * u_ins.t21[None] + m22 * (1.0 - u_ins.t21[None])
            generate_edge("Nearest", tm1, tm2)
            print("[taichi] linear t1,t2:{},{}".format(
                u_ins.t11[None], u_ins.t21[None]))
            print("[taichi] minimum distance:{}".format(
                surface_distance(tm1, tm2)))
        elif unit_test_selected == "Cone-Slab":
            u_ins.unit_cone_slab(m11, m12, m21, m22, m23)
            tm1 = m11 * u_ins.t11[None] + m12 * (1.0 - u_ins.t11[None])
            tm2 = m21 * u_ins.t21[None] + m22 * u_ins.t22[None] + \
                m23 * (1.0 - u_ins.t21[None] - u_ins.t22[None])
            generate_edge("Nearest", tm1, tm2)
            print("[taichi] linear t:{} \t barycentric t1,t2:{},{}".format(
                u_ins.t11[None], u_ins.t21[None], u_ins.t22[None]))
            print("[taichi] minimum distance:{}".format(
                surface_distance(tm1, tm2)))
        else:
            pass

    psimgui.PushItemWidth(150)
    _, steps = psimgui.InputInt("Steps", steps, step=1, step_fast=100)
    psimgui.PopItemWidth()
    psimgui.SameLine()
    if psimgui.Button("Check"):
        if unit_test_selected == "Sphere-Cone":
            pass
        elif unit_test_selected == "Sphere-Slab":
            u_ins.proof_sphere_slab(m11, m21, m22, m23, steps)
            print(u_ins.min_dis[None])
        elif unit_test_selected == "Cone-Cone":
            u_ins.proof_cone_cone(m11, m12, m21, m22, steps)
            print(u_ins.min_dis[None])
        elif unit_test_selected == "Cone-Slab":
            pass
        else:
            pass


ps.init()
ps.set_user_callback(callback)
ps.show()
