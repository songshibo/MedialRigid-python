import taichi as ti
import numpy as np
import taichi_glsl as ts

# ti.init(arch=ti.gpu, debug=True, kernel_profiler=True)


@ti.func
def linear_lerp(v1: ti.template(), v2: ti.template(), t: ti.f32) -> ti.template():
    return v1 * t + v2 * (1.0-t)


@ti.func
def bary_lerp(v1: ti.template(), v2: ti.template(), v3: ti.template(), t1: ti.f32, t2: ti.f32) -> ti.template():
    return v1 * t1 + v2 * t2 + v3 * (1.0 - t1 - t2)


@ti.func
def surface_distane(m1: ti.template(), m2: ti.template()) -> ti.f32:
    c1 = ti.Vector([m1.x, m1.y, m1.z])
    c2 = ti.Vector([m2.x, m2.y, m2.z])
    return ts.length(c1-c2) - (m1.w + m2.w)


# Check point is inside triangle
# this function does not check if P is coplanar with ABC
@ti.func
def is_point_inside_triangle(P, A, B, C):
    ab, ac, ap, pc, pb = B-A, C-A, P-A, C-P, B-P
    abc = ts.length(ts.cross(ab, ac))
    abp = ts.length(ts.cross(ab, ap))
    bcp = ts.length(ts.cross(pc, pb))
    cap = ts.length(ts.cross(ac, pc))
    t1, t2, t3 = bcp / abc, cap / abc, abp / abc
    return t1+t2+t3 == 1.0, t1, t2


# Check intersection of line segment and triangle
# segment: {e1,e2}, triangle: {A,B,C}
@ti.func
def is_segement_intersect_with_triangle(e1, e2, A, B, C):
    t1, t2, t3 = 0.0, 0.0, 0.0
    seg = e1-e2
    norm = ts.cross(A-C, B-C)
    v = ts.dot(seg, norm)
    intersected = False
    if v != 0.0:
        e2A = A - e2
        # linear interpolation parameter
        t1 = ts.dot(e2A, norm) / max(v, 1e-10)
        if (t1 < 0.0) or (t1 > 1.0):  # check if point is on segment
            intersected = False
        else:
            point = e2 + t1 * seg
            intersected, t2, t3 = is_point_inside_triangle(point, A, B, C)

    return intersected, t1, t2, t3


@ti.func
def get_sphere_cone_nearest(m: ti.template(),  m1: ti.template(), m2: ti.template()):
    cm = ti.Vector([m.x, m.y, m.z])
    c1 = ti.Vector([m1.x, m1.y, m1.z])
    c2 = ti.Vector([m2.x, m2.y, m2.z])
    r1, r2 = ti.static(m1.w, m2.w)
    inversed = False
    if r1 > r2:
        c1, c2 = c2, c1
        r1, r2 = r2, r1
        inversed = True

    c21 = c1 - c2
    cq2 = c2 - cm
    # S_c = A * t^2.0 + D * t + F
    A = ts.dot(c21, c21)
    D = 2.0 * ts.dot(c21, cq2)
    F = ts.dot(cq2, cq2)
    # S_r = R1 * t + R2
    R1 = r1 - r2

    t = -(A*D-R1*R1*D) - ti.sqrt((D*D-4.0*A*F)*(R1*R1-A)*R1*R1)
    t /= 2.0 * (A*A - A*R1*R1)
    t = ts.clamp(t, 0.0, 1.0)

    if inversed:
        t = 1.0-t

    return t


@ti.func
def get_sphere_slab_nearest(m: ti.template(), m1: ti.template(), m2: ti.template(), m3: ti.template()):
    cm = ti.Vector([m.x, m.y, m.z])
    c1 = ti.Vector([m1.x, m1.y, m1.z])
    c2 = ti.Vector([m2.x, m2.y, m2.z])
    c3 = ti.Vector([m3.x, m3.y, m3.z])
    r1, r2, r3 = ti.static(m1.w, m2.w, m3.w)
    inversed13, inversed23 = False, False
    if r1 > r3:
        c1, c3 = c3, c1
        r1, r3 = r3, r1
        inversed13 = True
    if r2 > r3:
        c2, c3 = c3, c2
        r2, r3 = r3, r2
        inversed23 = True

    c31 = c1-c3
    c32 = c2-c3
    cm3 = c3-cm
    R1 = r1-r3
    R2 = r2-r3

    A = ts.dot(c31, c31)
    B = 2.0 * ts.dot(c31, c32)
    C = ts.dot(c32, c32)
    D = 2.0 * ts.dot(c31, cm3)
    E = 2.0 * ts.dot(c32, cm3)
    F = ts.dot(cm3, cm3)

    t1, t2 = 0.0, 0.0
    if R1 == 0.0 and R2 == 0.0:
        denom = (4.0*A*C-B*B)
        t1 = (B*E - 2.0*C*D)/denom
        t2 = (B*D - 2.0*A*E)/denom
    elif R1 != 0.0 and R2 == 0.0:
        H2 = -B / (2.0 * C)
        K2 = -E / (2.0 * C)
        W1 = pow(2.0 * A + B * H2, 2.0) - 4.0 * R1 * \
            R1 * (A + B * H2 + C * H2 * H2)
        W2 = 2.0 * (2.0 * A + B * H2) * (B * K2 + D) - 4.0 * R1 * \
            R1 * (B * K2 + 2.0 * C * H2 * K2 + D + E * H2)
        W3 = pow(B * K2 + D, 2.0) - 4.0 * R1 * \
            R1 * (C * K2 * K2 + E * K2 + F)
        t1 = (-W2 - ti.sqrt(W2 * W2 - 4.0 * W1 * W3)) / (2.0 * W1)
        t2 = H2 * t1 + K2
    elif R1 == 0.0 and R2 != 0.0:
        H1 = -B / (2.0 * A)
        K1 = -D / (2.0 * A)
        W1 = pow(2.0 * C + B * H1, 2.0) - 4.0 * R2 * \
            R2 * (C + B * H1 + A * H1 * H1)
        W2 = 2.0 * (2.0 * C + B * H1) * (B * K1 + E) - 4.0 * R2 * \
            R2 * (B * K1 + 2.0 * A * H1 * K1 + E + D * H1)
        W3 = pow(B * K1 + E, 2.0) - 4.0 * R2 * \
            R2 * (A * K1 * K1 + D * K1 + F)
        t2 = (-W2 - ti.sqrt(W2 * W2 - 4.0 * W1 * W3)) / (2.0 * W1)
        t1 = H1 * t2 + K1
    else:
        L1 = 2.0 * A * R2 - B * R1
        L2 = 2.0 * C * R1 - B * R2
        L3 = E * R1 - D * R2
        if L1 == 0.0 and L2 != 0.0:
            t2 = -L3 / L2
            W1 = 4.0 * A * A - 4.0 * R1 * R1 * A
            W2 = 4.0 * A * (B * t2 + D) - 4.0 * R1 * R1 * (B * t2 + D)
            W3 = pow(B * t2 + D, 2.0) - (C * t2 * t2 + E * t2 + F)
            t1 = (-W2 - ti.sqrt(W2 * W2 - 4.0 * W1 * W3)) / (2.0 * W1)
        elif L1 != 0.0 and L2 == 0.0:
            t1 = L3 / L1
            W1 = 4.0 * C * C - 4.0 * R2 * R2 * C
            W2 = 4.0 * C * (B * t1 + E) - 4.0 * R2 * R2 * (B * t1 + E)
            W3 = pow(B * t1 + E, 2.0) - (A * t1 * t1 + D * t1 + F)
            t2 = (-W2 - ti.sqrt(W2 * W2 - 4.0 * W1 * W3)) / (2.0 * W1)
        else:
            H3 = L2 / L1
            K3 = L3 / L1
            W1 = pow((2.0 * C + B * H3), 2.0) - 4.0 * R2 * \
                R2 * (A * H3 * H3 + B * H3 + C)
            W2 = 2.0 * (2.0 * C + B * H3) * (B * K3 + E) - 4.0 * R2 * \
                R2 * (2.0 * A * H3 * K3 + B * K3 + D * H3 + E)
            W3 = pow((B * K3 + E), 2.0) - 4.0 * R2 * \
                R2 * (A * K3 * K3 + D * K3 + F)
            t2 = (-W2 - ti.sqrt(W2 * W2 - 4.0 * W1 * W3)) / (2.0 * W1)
            t1 = H3 * t2 + K3

    if (t1+t2) < 1.0 and 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0:  # nearest sphere is inside triangle
        # post-processing t1,t2
        if inversed13 and not inversed23:
            t1 = 1-t1-t2
        elif not inversed13 and inversed23:
            t2 = 1-t1-t2
        else:
            t1, t2 = t2, t1
            t2 = 1 - t1 - t2
    else:  # nearest sphere is on edge
        # first init with cone:{m1,m3}
        t = get_sphere_cone_nearest(m, m1, m3)
        t1, t2 = t, 0.0
        min_dis = surface_distane(m, linear_lerp(m1, m3, t))
        # cone:{m2,m3}
        t = get_sphere_cone_nearest(m, m2, m3)
        dis = surface_distane(m, linear_lerp(m2, m3, t))
        if dis < min_dis:
            min_dis = dis
            t1, t2 = 0.0, t
        # cone:{m1,m3}
        t = get_sphere_cone_nearest(m, m1, m2)
        dis = surface_distane(m, linear_lerp(m1, m2, t))
        if dis < min_dis:
            t1, t2 = t, 1.0-t

    return t1, t2


@ti.func
def get_cone_cone_nearest(m11: ti.template(), m12: ti.template(), m21: ti.template(), m22: ti.template()):
    c11 = ti.Vector([m11.x, m11.y, m11.z])
    c12 = ti.Vector([m12.x, m12.y, m12.z])
    c21 = ti.Vector([m21.x, m21.y, m21.z])
    c22 = ti.Vector([m22.x, m22.y, m22.z])
    r11, r12, r21, r22 = ti.static(m11.w, m12.w, m21.w, m22.w)

    inversed1, inversed2 = False, False
    if r11 > r12:
        c11, c12 = c12, c11
        r11, r12 = r12, r11
        inversed1 = True
    if r21 > r22:
        c21, c22 = c22, c21
        r21, r22 = r22, r21
        inversed2 = True

    c12c11 = c11-c12
    c22c21 = c21-c22
    c22c12 = c12-c22

    # S(x,y) = S_c(x,y)^(1/2) - S_r(x,y)
    # = [Ax^2+Bxy+Cy^2+Dx+Ey+F]^(1/2) - (R1x+R2y+R3)
    R1 = r11 - r12
    R2 = r21 - r22
    A = ts.dot(c12c11, c12c11)
    B = -2.0 * ts.dot(c12c11, c22c21)
    C = ts.dot(c22c21, c22c21)
    D = 2.0 * ts.dot(c12c11, c22c12)
    E = -2.0 * ts.dot(c22c21, c22c12)
    F = ts.dot(c22c12, c22c12)

    t1, t2 = 0.0, 0.0
    # cone:{m11,m12} parallel to cone:{m21,m22}
    if (4.0 * A * C - B * B) == 0.0:
        # find minimum distance
        # m11 to cone:{m21,m22}
        t = get_sphere_cone_nearest(m11, m21, m22)
        min_dis = surface_distane(m11, linear_lerp(m21, m22, t))
        t1, t2 = 1.0, t
        # m12 to cone:{m21,m22}
        t = get_sphere_cone_nearest(m12, m21, m22)
        dis = surface_distane(m12, linear_lerp(m21, m22, t))
        if dis < min_dis:
            t1, t2 = 0.0, t
            min_dis = dis
        # m21 to cone:{m11,m12}
        t = get_sphere_cone_nearest(m21, m11, m12)
        dis = surface_distane(m21, linear_lerp(m11, m12, t))
        if dis < min_dis:
            t1, t2 = t, 1.0
            min_dis = dis
        # m22 to cone:{m11,m12}
        t = get_sphere_cone_nearest(m22, m11, m12)
        dis = surface_distane(m22, linear_lerp(m11, m12, t))
        if dis < min_dis:
            t1, t2 = t, 0.0
    else:
        if R1 != 0.0 and R2 != 0.0:
            L1 = 2.0 * A * R2 - B * R1
            L2 = 2.0 * C * R1 - B * R2
            L3 = E * R1 - D * R2
            # L1=0,L2=0 has already been considered
            if L1 == 0.0 and L2 != 0.0:
                t2 = -1.0 * L3 / L2
                W1 = 4.0 * A * A - 4.0 * R1 * R1 * A
                W2 = 4.0 * A * (B * t2 + D) - 4.0 * R1 * R1 * (B * t2 + D)
                W3 = (B * t2 + D) * (B * t2 + D) - 4.0 * \
                    R1 * R1 * (C * t2 * t2 + E * t2 + F)
                if W1 == 0.0:
                    t1 = -W3 / W2
                else:
                    delta = max(W2 * W2 - 4.0 * W1 * W3, 0.0)
                    t1 = (-W2 - ti.sqrt(delta)) / (2.0 * W1)
            elif L1 != 0.0 and L2 == 0.0:
                t1 = L3 / L1
                W1 = 4.0 * C * C - 4.0 * R2 * R2 * C
                W2 = 4.0 * C * (B * t1 + E) - 4.0 * R2 * R2 * (B * t1 + E)
                W3 = (B * t1 + E) * (B * t1 + E) - 4.0 * \
                    R2 * R2 * (A * t1 * t1 + D * t1 + F)
                if W1 == 0.0:
                    t2 = -W3 / W2
                else:
                    delta = max(W2 * W2 - 4.0 * W1 * W3, 0.0)
                    t2 = (-W2 - ti.sqrt(delta)) / (2.0 * W1)
            else:  # L1,L2 != 0
                H3 = L2 / L1
                K3 = L3 / L1
                W1 = pow(2.0 * C + B * H3, 2.0) - 4.0 * R2 * \
                    R2 * (A * H3 * H3 + B * H3 + C)
                W2 = 2.0 * (2.0 * C + B * H3) * (B * K3 + E) - 4.0 * \
                    R2 * R2 * (2.0 * A * H3 * K3 + B * K3 + D * H3 + E)
                W3 = pow(B * K3 + E, 2.0) - 4.0 * R2 * \
                    R2 * (A * K3 * K3 + D * K3 + F)
                delta = max(W2 * W2 - 4.0 * W1 * W3, 0.0)
                t2 = (-W2 - ti.sqrt(delta)) / (2.0 * W1)
                t1 = H3 * t2 + K3
        elif R1 == 0.0 and R2 == 0.0:
            denom = 4.0 * A * C - B * B
            t1 = (B * E - 2.0 * C * D) / denom
            t2 = (B * E - 2.0 * A * E) / denom
        elif R1 == 0.0 and R2 != 0.0:
            H1 = -B / (2.0 * A)
            K1 = -D / (2.0 * A)
            W1 = pow(2.0 * C + B * H1, 2.0) - 4.0 * R2 * \
                R2 * (A * H1 * H1 + B * H1 + C)
            W2 = 2.0 * (2.0 * C + B * H1) * (B * K1 + E) - 4.0 * R2 * \
                R2 * (2.0 * A * H1 * K1 + B * K1 + D * H1 + E)
            W3 = pow(B * K1 + E, 2.0) - 4.0 * R2 * \
                R2 * (A * K1 * K1 + D * K1 + F)
            delta = max(W2 * W2 - 4.0 * W1 * W3, 0.0)
            t2 = (-W2 - ti.sqrt(delta)) / (2.0 * W1)
            t1 = H1 * t2 + K1
        else:  # R1 != 0 and R2 == 0
            H2 = -B / (2.0 * C)
            K2 = -E / (2.0 * C)
            W1 = pow(2.0 * A + B * H2, 2) - 4.0 * R1 * \
                R1 * (A + B * H2 + C * H2 * H2)
            W2 = 2.0 * (2.0 * A + B * H2) * (B * K2 + D) - 4.0 * R1 * \
                R1 * (B * K2 + 2.0 * C * H2 * K2 + D + E * H2)
            W3 = pow(B * K2 + D, 2) - 4.0 * R1 * \
                R1 * (C * K2 * K2 + E * K2 + F)
            delta = max(W2 * W2 - 4.0 * W1 * W3, 0.0)
            t1 = (-W2 - ti.sqrt(delta)) / (2.0 * W1)
            t2 = H2 * t1 + K2

        # post-processing t1,t2
        if inversed1:
            t1 = 1.0-t1
        if inversed2:
            t2 = 1.0-t2
        if 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0:
            pass
        else:
            t1, t2 = ts.clamp(t1, 0.0, 1.0), ts.clamp(t2, 0.0, 1.0)
            l_m1 = linear_lerp(m11, m12, t1)
            l_m2 = linear_lerp(m21, m22, t2)
            # find minimum distance
            t21 = get_sphere_cone_nearest(l_m1, m21, m22)
            d1 = surface_distane(l_m1, linear_lerp(m21, m22, t21))
            t11 = get_sphere_cone_nearest(l_m2, m11, m12)
            d2 = surface_distane(l_m2, linear_lerp(m11, m12, t11))
            if d1 < d2:
                t2 = t21
            else:
                t1 = t11

    return t1, t2


@ti.func
def get_cone_slab_nearest(m11: ti.template(), m12: ti.template(), m21: ti.template(), m22: ti.template(), m23: ti.template()):
    c11 = ti.Vector([m11.x, m11.y, m11.z])
    c12 = ti.Vector([m12.x, m12.y, m12.z])
    c21 = ti.Vector([m21.x, m21.y, m21.z])
    c22 = ti.Vector([m22.x, m22.y, m22.z])
    c23 = ti.Vector([m23.x, m23.y, m23.z])

    intersected, t1, t2, t3 = is_segement_intersect_with_triangle(
        c11, c12, c21, c22, c23)
    if not intersected:
        # Cone to Cone
        # initial as {m11,m12} to {m21,m22}
        t1, t2 = get_cone_cone_nearest(m11, m12, m21, m22)
        t3 = 1.0 - t2
        min_dis = surface_distane(linear_lerp(
            m11, m12, t1), linear_lerp(m21, m22, t2))
        # {m11,m12} to {m21,m23}
        tt1, tt2 = get_cone_cone_nearest(m11, m12, m21, m23)
        dis = surface_distane(linear_lerp(m11, m12, tt1),
                              linear_lerp(m21, m23, tt2))
        if dis < min_dis:
            min_dis = dis
            t1, t2, t3 = tt1, tt2, 0.0
        # {m11,m12} to {m22,m23}
        tt1, tt2 = get_cone_cone_nearest(m11, m12, m22, m23)
        dis = surface_distane(linear_lerp(m11, m12, tt1),
                              linear_lerp(m22, m23, tt2))
        if dis < min_dis:
            min_dis = dis
            t1, t2, t3 = tt1, 0.0, tt2
        # Sphere to Slab
        # m11 to {m21,m22,m23}
        tt1, tt2 = get_sphere_slab_nearest(m11, m21, m22, m23)
        dis = surface_distane(m11, bary_lerp(m21, m22, m23, tt1, tt2))
        if dis < min_dis:
            min_dis = dis
            t1, t2, t3 = 1.0, tt1, tt2
        # m12 to {m21,m22,m23}
        tt1, tt2 = get_sphere_slab_nearest(m12, m21, m22, m23)
        dis = surface_distane(m12, bary_lerp(m21, m22, m23, tt1, tt2))
        if dis < min_dis:
            t1, t2, t3 = 0.0, tt1, tt2

    return t1, t2, t3


cq = ti.Vector([0.0, 0.5, 0.5, 0.3])
cone_m1 = ti.Vector([0.5, 0.0, 0.0, 0.35])
cone_m2 = ti.Vector([-0.5, 0.0, 0.0, 0.5])
# cone_m3 = ti.Vector([0.5, 0.0, 1.0, 0.2])
# cone_m4 = ti.Vector([0.0, 0.0, 0.0, 0.15])

# s = ti.field(ti.f32, shape=())
# s[None] = 0


@ti.data_oriented
class UnitTest:
    def __init__(self):
        self.t11 = ti.field(ti.f32, shape=())
        self.t12 = ti.field(ti.f32, shape=())
        self.t21 = ti.field(ti.f32, shape=())
        self.t22 = ti.field(ti.f32, shape=())
        self.min_dis = ti.field(ti.f32, shape=())

    @ti.kernel
    def unit_sphere_cone(self, m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr()):
        ti_m1 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        ti_m2 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        ti_m3 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        self.t21[None] = get_sphere_cone_nearest(ti_m1, ti_m2, ti_m3)

    @ti.kernel
    def unit_cone_cone(self, m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr(), m4: ti.any_arr()):
        ti_m1 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        ti_m2 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        ti_m3 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        ti_m4 = ti.Vector([m4[0], m4[1], m4[2], m4[3]])
        self.t11[None], self.t21[None] = get_cone_cone_nearest(
            ti_m1, ti_m2, ti_m3, ti_m4)

    @ti.kernel
    def unit_sphere_slab(self, m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr(), m4: ti.any_arr()):
        ti_m1 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        ti_m2 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        ti_m3 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        ti_m4 = ti.Vector([m4[0], m4[1], m4[2], m4[3]])
        self.t21[None], self.t22[None] = get_sphere_slab_nearest(
            ti_m1, ti_m2, ti_m3, ti_m4)

    @ti.kernel
    def unit_cone_slab(self, m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr(), m4: ti.any_arr(), m5: ti.any_arr()):
        ti_m1 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        ti_m2 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        ti_m3 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        ti_m4 = ti.Vector([m4[0], m4[1], m4[2], m4[3]])
        ti_m5 = ti.Vector([m5[0], m5[1], m5[2], m5[3]])
        self.t11[None], self.t21[None], self.t22[None] = get_cone_slab_nearest(
            ti_m1, ti_m2, ti_m3, ti_m4, ti_m5)

    @ti.kernel
    def proof_sphere_slab(self, m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr(), m4: ti.any_arr()):
        ti_m1 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        ti_m2 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        ti_m3 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        ti_m4 = ti.Vector([m4[0], m4[1], m4[2], m4[3]])
        self.min_dis[None] = 1000000.0
        min_t1, min_t2 = -1.0, -1.0
        for i, j in ti.ndrange(5000, 5000):
            t1, t2 = i / 5000.0, j / 5000.0
            if t1 + t2 <= 1.0:
                m = bary_lerp(ti_m2, ti_m3, ti_m4, t1, t2)
                dis = surface_distane(ti_m1, m)
                if dis < self.min_dis[None]:
                    ti.atomic_min(self.min_dis[None], dis)
                    min_t1, min_t2 = t1, t2
        print("{},{}".format(min_t1, min_t2))


@ti.kernel
def unit_test():
    t = get_sphere_cone_nearest(cq, cone_m1, cone_m2)
    print(t)
    # t1, t2 = get_cone_cone_nearest(cq, cone_m1, cone_m2, cone_m3)
    # print(get_sphere_cone_nearest(cq, cone_m1, cone_m2))
    # t1, t2 = get_sphere_slab_nearest(cq, cone_m1, cone_m2, cone_m3)
    # t1, t2, t3 = get_cone_slab_nearest(cq, cone_m1, cone_m2, cone_m3, cone_m4)
    # print("t1:{},t2:{},t3:{}".format(t1, t2, t3))
    # min_dis = surface_distane(cq, bary_lerp(cone_m1, cone_m2, cone_m3, t1, t2))...............................................................
    # min_dis = surface_distane(linear_lerp(
    #     cq, cone_m1, t1), linear_lerp(cone_m2, cone_m3, t2))
    # min_dis = surface_distane(linear_lerp(cq, cone_m1, t1), bary_lerp(
    #     cone_m2, cone_m3, cone_m4, t2, t3))
    # print(min_dis)
    # for i, j in ti.ndrange(1000, 1000):
    #     t1, t2 = get_sphere_slab_nearest(cq, cone_m1, cone_m2, cone_m3)
    #     min_dis = surface_distane(cq, bary_lerp(
    #         cone_m1, cone_m2, cone_m3, t1, t2))
    #     ti.atomic_max(s[None], min_dis)
    # print(s[None])

    # unit_test()
    # ti.print_kernel_profile_info()


# unit_test()
