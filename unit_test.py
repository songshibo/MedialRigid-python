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
    scale = np.ones(3) * _scale
    S = scale_matrix(scale)
    mesh.set_transform(concatenate_matrices(T, S))


def register_sphere(name, m, v, f, color, enable=True):
    mesh = ps.register_surface_mesh(name, v, f)
    TS(mesh, m[:3], m[3] * 2.0)
    mesh.set_color(color)
    mesh.set_enabled(enable)
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
    pos = m[0:3, 3]
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


def callback():
    global m11, m12, m13, m21, m22, m23, unit_test_selected

    selected_name = ps.get_selection()[0]
    if selected_name == "m11":
        m11 = decom_transform(ps.get_surface_mesh(
            selected_name).get_transform())
        update_curve_network()
    elif selected_name == "m12":
        m12 = decom_transform(ps.get_surface_mesh(
            selected_name).get_transform())
        update_curve_network()
    elif selected_name == "m21":
        m21 = decom_transform(ps.get_surface_mesh(
            selected_name).get_transform())
        update_curve_network()
    elif selected_name == "m22":
        m22 = decom_transform(ps.get_surface_mesh(
            selected_name).get_transform())
        update_curve_network()
    elif selected_name == "m23":
        m23 = decom_transform(ps.get_surface_mesh(
            selected_name).get_transform())
        update_curve_network()
    else:
        pass

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
        elif unit_test_selected == "Sphere-Slab":
            u_ins.unit_sphere_slab(m11, m21, m22, m23)
            tm = m21 * u_ins.t21[None] + m22 * u_ins.t22[None] + \
                m23 * (1.0 - u_ins.t21[None] - u_ins.t22[None])
            generate_edge("Nearest", m11, tm)
            print(surface_distance(m11, tm))
        elif unit_test_selected == "Cone-Cone":
            u_ins.unit_cone_cone(m11, m12, m21, m22)
            tm1 = m11 * u_ins.t11[None] + m12 * (1.0 - u_ins.t11[None])
            tm2 = m21 * u_ins.t21[None] + m22 * (1.0 - u_ins.t21[None])
            generate_edge("Nearest", tm1, tm2)
        elif unit_test_selected == "Cone-Slab":
            u_ins.unit_cone_slab(m11, m12, m21, m22, m23)
            tm1 = m11 * u_ins.t11[None] + m12 * (1.0 - u_ins.t11[None])
            tm2 = m21 * u_ins.t21[None] + m22 * u_ins.t22[None] + \
                m23 * (1.0 - u_ins.t21[None] - u_ins.t22[None])
            generate_edge("Nearest", tm1, tm2)
        else:
            pass

    if psimgui.Button("Check"):
        u_ins.proof_sphere_slab(m11, m21, m22, m23)
        print(u_ins.min_dis[None])


ps.init()
ps.set_user_callback(callback)
ps.show()
