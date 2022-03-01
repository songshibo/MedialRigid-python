import polyscope as ps
import polyscope.imgui as psimgui
import os

from engine.simulator import Simulator
from util import *
from engine import *
import taichi as ti

ti.init(arch=ti.gpu, debug=False,
        advanced_optimization=False, kernel_profiler=True)


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


def generate_edge_with_v(name, m1, m2, v1, v2):
    v = np.array([m1[:3], m2[:3]])
    e = np.array([[0, 1]])
    curve = ps.register_curve_network(name, v, e)
    vecs_node = np.array([v1, v2])
    curve.add_vector_quantity(
        "v-edge", vecs_node, enabled=True, length=1.0, radius=0.02)


def generate_triangle(name, m1, m2, m3):
    v = np.array([m1[:3], m2[:3], m3[:3]])
    e = np.array([[0, 1], [1, 2], [2, 0]])
    ps.remove_curve_network(name, error_if_absent=False)
    ps.register_curve_network(name, v, e)


def generate_triangle_with_v(name, m1, m2, m3, v1, v2, v3):
    v = np.array([m1[:3], m2[:3], m3[:3]])
    e = np.array([[0, 1], [1, 2], [2, 0]])
    ps.remove_curve_network(name, error_if_absent=False)
    curve = ps.register_curve_network(name, v, e)
    vecs_nodes = np.array([v1, v2, v3])
    curve.add_vector_quantity("v-edge", vecs_nodes,
                              enabled=True, length=1.0, radius=0.02)


vs, fs = igl.read_triangle_mesh("./models/unit_sphere.obj")

unit_test_options = ["Sphere-Cone", "Sphere-Slab",
                     "Cone-Cone", "Cone-Slab", "Slab-Slab"]
unit_test_selected = unit_test_options[0]


def update_curve_network():
    global unit_test_selected
    if unit_test_selected == "Sphere-Cone":
        ps.remove_curve_network("p1", error_if_absent=False)
        generate_edge_with_v("p2", m21, m22, v11, v12)
    elif unit_test_selected == "Sphere-Slab":
        ps.remove_curve_network("p1", error_if_absent=False)
        generate_triangle_with_v("p2", m21, m22, m23, v21, v22, v23)
    elif unit_test_selected == "Cone-Cone":
        generate_edge_with_v("p1", m11, m12, v11, v12)
        generate_edge_with_v("p2", m21, m22, v21, v22)
    elif unit_test_selected == "Cone-Slab":
        generate_edge_with_v("p1", m11, m12, v11, v12)
        generate_triangle_with_v("p2", m21, m22, m23, v21, v22, v23)
    else:
        generate_triangle_with_v("p1", m11, m12, m13, v11, v12, v13)
        generate_triangle_with_v("p2", m21, m22, m23, v21, v22, v23)


def decom_transform(m):
    pos = np.array(m[0:3, 3])
    m[0:3, 3] = np.zeros(3)
    M_without_rot = np.matmul(m, m.T)
    scale = np.sqrt(M_without_rot[0, 0]) * 0.5
    return np.array([pos[0], pos[1], pos[2], scale])


# first
m11 = np.array([0.0, 0.7, 0.5, 0.3])
v11 = np.array([0.0, -1.0, 0.0])
m12 = np.array([0.0, 1.0, 0.0, 0.35])
v12 = np.array([0.0, -0.5, 0.3])
m13 = np.array([0.5, 0.7, -0.4, 0.25])
v13 = np.array([0.0, -1.0, 0.0])
# second
m21 = np.array([0.5, 0.0, 0.0, 0.35])
v21 = np.array([0.0, 0.0, 0.0])
m22 = np.array([-0.5, 0.0, 0.0, 0.5])
v22 = np.array([0.0, 0.0, 0.0])
m23 = np.array([0.0, 0.0, -0.6, 0.2])
v23 = np.array([0.0, 0.0, 0.0])

color1 = np.array([235.0, 64.0, 52.0, 255.0]) / 255.0
color2 = np.array([52.0, 177.0, 235.0, 255.0]) / 255.0
sp11 = register_sphere("m11", m11, vs, fs, color1, True)
sp12 = register_sphere("m12", m12, vs, fs, color1, False)
sp13 = register_sphere("m13", m13, vs, fs, color1, False)
# first medial primitive lines

sp21 = register_sphere("m21", m21, vs, fs, color2, True)
sp22 = register_sphere("m22", m22, vs, fs, color2, True)
sp23 = register_sphere("m23", m23, vs, fs, color2, False)
# second medial primitive lines
generate_edge_with_v("p2", m21, m22, v21, v22)

u_ins = UnitTest()
steps = 5000
global_t = 1.0


def callback():
    global m11, m12, m13, m21, m22, m23, v11, v12, v13, v21, v22, v23, global_t, unit_test_selected, steps

    if psimgui.Button("Update from Scene"):
        m11 = decom_transform(ps.get_surface_mesh(
            "m11").get_transform())
        m12 = decom_transform(ps.get_surface_mesh(
            "m12").get_transform())
        m13 = decom_transform(ps.get_surface_mesh(
            "m13").get_transform())
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
        sp13.set_transparency(0.5)
        sp21.set_transparency(0.5)
        sp22.set_transparency(0.5)
        sp23.set_transparency(0.5)
    psimgui.SameLine()
    if psimgui.Button("Make Spheres Solid"):
        sp11.set_transparency(1)
        sp12.set_transparency(1)
        sp13.set_transparency(1)
        sp21.set_transparency(1)
        sp22.set_transparency(1)
        sp23.set_transparency(1)

    psimgui.PushItemWidth(200)
    changed = psimgui.BeginCombo("Unit Test Type", unit_test_selected)
    if changed:
        for val in unit_test_options:
            changed, selected = psimgui.Selectable(
                val, unit_test_selected == val)
            if selected and changed:
                unit_test_selected = val
                update_curve_network()
        psimgui.EndCombo()
    psimgui.PopItemWidth()
    psimgui.Separator()

    psimgui.TextUnformatted("First Primitive")
    changed, m11 = psimgui.InputFloat4("Medial Sphere11", m11)
    if changed:
        update_curve_network()
        TS(sp11, m11[:3], m11[3])
    if unit_test_selected == "Cone-Cone" or unit_test_selected == "Cone-Slab" or unit_test_selected == "Slab-Slab":
        sp12.set_enabled(True)
        changed, m12 = psimgui.InputFloat4("Medial Sphere12", m12)
        if changed:
            update_curve_network()
            TS(sp12, m12[:3], m12[3])
    else:
        sp12.set_enabled(False)
    if unit_test_selected == "Slab-Slab":
        sp13.set_enabled(True)
        changed, m13 = psimgui.InputFloat4("Medial Sphere13", m13)
        if changed:
            update_curve_network()
            TS(sp13, m13[:3], m13[3])
    else:
        sp13.set_enabled(False)

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
    if unit_test_selected == "Sphere-Slab" or unit_test_selected == "Cone-Slab" or unit_test_selected == "Slab-Slab":
        sp23.set_enabled(True)
        changed, m23 = psimgui.InputFloat4("Medial Sphere23", m23)
        if changed:
            update_curve_network()
            TS(sp23, m23[:3], m23[3])
    else:
        sp23.set_enabled(False)

    psimgui.Separator()
    m11, m12, m13, m21, m22, m23 = np.array(m11, dtype=np.float32), np.array(m12, dtype=np.float32), np.array(
        m13, dtype=np.float32), np.array(m21, dtype=np.float32), np.array(m22, dtype=np.float32), np.array(m23, dtype=np.float32)
    if psimgui.Button("Compute Neareset"):
        if unit_test_selected == "Sphere-Cone":
            u_ins.unit_sphere_cone(m11, m21, m22)
            tm = m21 * u_ins.t21[None] + m22 * (1.0 - u_ins.t21[None])
            generate_edge("Nearest", m11, tm)
            print("[Medial-Rigid] linear t:{}".format(u_ins.t21[None]))
            print("[Medial-Rigid] minimum surface distance:{}".format(
                surface_distance(m11, tm)))
        elif unit_test_selected == "Sphere-Slab":
            u_ins.unit_sphere_slab(m11, m21, m22, m23)
            tm = m21 * u_ins.t21[None] + m22 * u_ins.t22[None] + \
                m23 * (1.0 - u_ins.t21[None] - u_ins.t22[None])
            generate_edge("Nearest", m11, tm)
            print("[Medial-Rigid] barycentric t1,t2:{},{}".format(
                u_ins.t21[None], u_ins.t22[None]))
            print("[Medial-Rigid] minimum surface distance:{}".format(
                surface_distance(m11, tm)))
        elif unit_test_selected == "Cone-Cone":
            u_ins.unit_cone_cone(m11, m12, m21, m22)
            tm1 = m11 * u_ins.t11[None] + m12 * (1.0 - u_ins.t11[None])
            tm2 = m21 * u_ins.t21[None] + m22 * (1.0 - u_ins.t21[None])
            generate_edge("Nearest", tm1, tm2)
            print("[Medial-Rigid] linear t1,t2:{},{}".format(
                u_ins.t11[None], u_ins.t21[None]))
            print("[Medial-Rigid] minimum surface distance:{}".format(
                surface_distance(tm1, tm2)))
        elif unit_test_selected == "Cone-Slab":
            u_ins.unit_cone_slab(m11, m12, m21, m22, m23)
            tm1 = m11 * u_ins.t11[None] + m12 * (1.0 - u_ins.t11[None])
            tm2 = m21 * u_ins.t21[None] + m22 * u_ins.t22[None] + \
                m23 * (1.0 - u_ins.t21[None] - u_ins.t22[None])
            generate_edge("Nearest", tm1, tm2)
            print("[Medial-Rigid] linear t:{} \t barycentric t1,t2:{},{}".format(
                u_ins.t11[None], u_ins.t21[None], u_ins.t22[None]))
            print("[Medial-Rigid] minimum surface distance:{}".format(
                surface_distance(tm1, tm2)))
        else:
            u_ins.unit_slab_slab(m11, m12, m13, m21, m22, m23)
            tm1 = m11 * u_ins.t11[None] + m12 * u_ins.t12[None] + \
                m13 * (1-u_ins.t11[None]-u_ins.t12[None])
            tm2 = m21 * u_ins.t21[None] + m22 * u_ins.t22[None] + \
                m23 * (1-u_ins.t21[None]-u_ins.t22[None])
            generate_edge("Nearest", tm1, tm2)
            print("[Medial-Rigid] barycentric t1,t2:{},{} \t barycentric t3,t4:{},{}".format(
                u_ins.t11[None], u_ins.t12[None], u_ins.t21[None], u_ins.t22[None]))
            print("[Medial-Rigid] minimum surface distance:{}".format(
                surface_distance(tm1, tm2)))

    psimgui.SameLine()
    if psimgui.Button("Intersection Test"):
        if unit_test_selected == "Sphere-Cone":
            ps.info(
                "Intersection test of Sphere-Cone is simply finding the neareset sphere on cone.")
        elif unit_test_selected == "Sphere-Slab":
            ps.info(
                "Intersection test of Sphere-Slab is simply finding the neareset sphere on slab.")
        elif unit_test_selected == "Cone-Cone":
            r = u_ins.unit_detect_cone_cone(m11, m12, m21, m22)
            print("Intersected:{}".format(r == 1))
        elif unit_test_selected == "Cone-Slab":
            r = u_ins.unit_detect_cone_slab(m11, m12, m21, m22, m23)
            print("Intersected:{}".format(r == 1))
        else:
            pass

    psimgui.PushItemWidth(150)
    _, steps = psimgui.InputInt("Steps", steps, step=1, step_fast=100)
    psimgui.PopItemWidth()
    psimgui.SameLine()
    if psimgui.Button("Check"):
        if unit_test_selected == "Sphere-Cone":
            u_ins.proof_sphere_cone(m11, m21, m22, steps)
        elif unit_test_selected == "Sphere-Slab":
            u_ins.proof_sphere_slab(m11, m21, m22, m23, steps)
        elif unit_test_selected == "Cone-Cone":
            u_ins.proof_cone_cone(m11, m12, m21, m22, steps)
        elif unit_test_selected == "Cone-Slab":
            u_ins.proof_cone_slab(m11, m12, m21, m22, m23, steps)
        else:
            u_ins.proof_slab_slab(m11, m12, m13, m21, m22, m23, steps)
        print("[Medial-Rigid] Searched minimum distance:{}".format(u_ins.min_dis[None]))

    psimgui.Separator()
    if unit_test_selected == "Cone-Cone":
        changed, v11 = psimgui.InputFloat3("V11", v11)
        if changed:
            generate_edge_with_v("p1", m11, m12, v11, v12)
        changed, v12 = psimgui.InputFloat3("V12", v12)
        if changed:
            generate_edge_with_v("p1", m11, m12, v11, v12)
        changed, v21 = psimgui.InputFloat3("V21", v21)
        if changed:
            generate_edge_with_v("p2", m21, m22, v21, v22)
        changed, v22 = psimgui.InputFloat3("V22", v22)
        if changed:
            generate_edge_with_v("p2", m21, m22, v21, v22)
    elif unit_test_selected == "Sphere-Slab":
        _, v11 = psimgui.InputFloat3("V11", v11)

        changed, v21 = psimgui.InputFloat3("V21", v21)
        if changed:
            generate_triangle_with_v("p2", m21, m22, m23, v21, v22, v23)
        changed, v22 = psimgui.InputFloat3("V22", v22)
        if changed:
            generate_triangle_with_v("p2", m21, m22, m23, v21, v22, v23)
        changed, v23 = psimgui.InputFloat3("V23", v23)
        if changed:
            generate_triangle_with_v("p2", m21, m22, m23, v21, v22, v23)
    else:
        changed, v11 = psimgui.InputFloat3("V11", v11)
        if changed:
            generate_triangle_with_v("p1", m11, m12, m13, v11, v12, v13)
        changed, v12 = psimgui.InputFloat3("V12", v12)
        if changed:
            generate_triangle_with_v("p1", m11, m12, m13, v11, v12, v13)
        changed, v13 = psimgui.InputFloat3("V13", v13)
        if changed:
            generate_triangle_with_v("p1", m11, m12, m13, v11, v12, v13)
        changed, v21 = psimgui.InputFloat3("V21", v21)
        if changed:
            generate_triangle_with_v("p2", m21, m22, m23, v21, v22, v23)
        changed, v22 = psimgui.InputFloat3("V22", v22)
        if changed:
            generate_triangle_with_v("p2", m21, m22, m23, v21, v22, v23)
        changed, v23 = psimgui.InputFloat3("V23", v23)
        if changed:
            generate_triangle_with_v("p2", m21, m22, m23, v21, v22, v23)

    v11 = np.array(v11).astype(np.float32)
    v12 = np.array(v12).astype(np.float32)
    v13 = np.array(v13).astype(np.float32)
    v21 = np.array(v21).astype(np.float32)
    v22 = np.array(v22).astype(np.float32)
    v23 = np.array(v23).astype(np.float32)
    if psimgui.Button("Find TOI"):
        if unit_test_selected == "Cone-Cone":
            toi = u_ins.moving_cone_cone(
                m11, m12, m21, m22, v11, v12, v21, v22)
            m11[:3] += v11 * toi
            m12[:3] += v12 * toi
            m21[:3] += v21 * toi
            m22[:3] += v22 * toi
            TS(sp11, m11[:3], m11[3])
            TS(sp12, m12[:3], m12[3])
            TS(sp21, m21[:3], m21[3])
            TS(sp22, m22[:3], m22[3])
            generate_edge("p1", m11, m12)
            generate_edge("p2", m21, m22)
        elif unit_test_selected == "Sphere-Slab":
            toi = u_ins.moving_sphere_slab(
                m11, m21, m22, m23, v11, v21, v22, v23)
            m11[:3] += v11 * toi
            m21[:3] += v21 * toi
            m22[:3] += v22 * toi
            m23[:3] += v23 * toi
            TS(sp11, m11[:3], m11[3])
            TS(sp21, m21[:3], m21[3])
            TS(sp22, m22[:3], m22[3])
            TS(sp23, m23[:3], m23[3])
        elif unit_test_selected == "Cone-Slab":
            toi = u_ins.moving_cone_slab(
                m11, m12, m21, m22, m23, v11, v12, v21, v22, v23)
            m11[:3] += v11 * toi
            m12[:3] += v12 * toi
            m21[:3] += v21 * toi
            m22[:3] += v22 * toi
            m23[:3] += v23 * toi
            TS(sp11, m11[:3], m11[3])
            TS(sp12, m12[:3], m12[3])
            TS(sp21, m21[:3], m21[3])
            TS(sp22, m22[:3], m22[3])
            TS(sp23, m23[:3], m23[3])
        elif unit_test_selected == "Slab-Slab":
            toi = u_ins.moving_slab_slab(
                m11, m12, m13, m21, m22, m23, v11, v12, v13, v21, v22, v23)
            m11[:3] += v11 * toi
            m12[:3] += v12 * toi
            m13[:3] += v13 * toi
            m21[:3] += v21 * toi
            m22[:3] += v22 * toi
            m23[:3] += v23 * toi
            TS(sp11, m11[:3], m11[3])
            TS(sp12, m12[:3], m12[3])
            TS(sp13, m13[:3], m13[3])
            TS(sp21, m21[:3], m21[3])
            TS(sp22, m22[:3], m22[3])
            TS(sp23, m23[:3], m23[3])

    psimgui.SameLine()
    if psimgui.Button("Performance of TOI"):
        if unit_test_selected == "Cone-Cone":
            u_ins.cone_cone_performance(
                m11, m12, m21, m22, v11, v12, v21, v22, steps)
            ti.print_kernel_profile_info('trace')
        elif unit_test_selected == "Slab-Slab":
            u_ins.slab_slab_performance(
                m11, m12, m13, m21, m22, m23, v11, v12, v13, v21, v22, v23, steps)
            ti.print_kernel_profile_info('trace')
        ti.clear_kernel_profile_info()

    _, global_t = psimgui.InputFloat(
        "advanced t", global_t, 0.0001, 1.0, "%.6f")
    if psimgui.Button("Advance"):
        m11[:3] += v11 * global_t
        m12[:3] += v12 * global_t
        m21[:3] += v21 * global_t
        m22[:3] += v22 * global_t
        m23[:3] += v23 * global_t
        TS(sp11, m11[:3], m11[3])
        TS(sp12, m12[:3], m12[3])
        TS(sp21, m21[:3], m21[3])
        TS(sp22, m22[:3], m22[3])
        TS(sp23, m23[:3], m23[3])


ps.init()
ps.set_user_callback(callback)
ps.show()
