import taichi as ti
import taichi_glsl as ts


@ti.func
def linear_lerp(v1: ti.template(), v2: ti.template(), t: ti.f32):
    return v1 * t + v2 * (1.0-t)


@ti.func
def bary_lerp(v1: ti.template(), v2: ti.template(), v3: ti.template(), t1: ti.f32, t2: ti.f32):
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
def is_segements_parallel_to_triangle(c12c11, c23c21, c23c22):
    c12c11_n = ts.normalize(c12c11)
    normal = ts.normalize(ts.cross(c23c21, c23c22))
    return ts.dot(c12c11_n, normal) == 0.0


# solve Ax^2 + Bx + C = 0
@ti.func
def solve_quadratic(A, B, C):
    has_solution, root1, root2 = True, -1.0, -1.0
    if A == 0.0:  # degnerated case
        if B != 0.0:
            root1 = -C / B
            root2 = root1
    else:
        delta = B*B - 4.0 * A * C
        if delta < 0.0:
            has_solution = False
        else:
            root1 = (-B - ti.sqrt(delta)) / (2 * A)
            root2 = (-B + ti.sqrt(delta)) / (2 * A)
    return has_solution, root1, root2


#######################################
## Finding Neareset Sphere Algorithm ##
#######################################
@ti.func
def get_sphere_cone_nearest(m: ti.template(),  m1: ti.template(), m2: ti.template()):
    cm = ti.Vector([m.x, m.y, m.z])
    c1 = ti.Vector([m1.x, m1.y, m1.z])
    c2 = ti.Vector([m2.x, m2.y, m2.z])
    r1, r2 = m1.w, m2.w
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
    # cm = ti.Vector([m.x, m.y, m.z])
    # c1 = ti.Vector([m1.x, m1.y, m1.z])
    # c2 = ti.Vector([m2.x, m2.y, m2.z])
    # c3 = ti.Vector([m3.x, m3.y, m3.z])
    # r1, r2, r3 = m1.w, m2.w, m3.w

    c31 = ti.Vector([m1.x-m3.x, m1.y-m3.y, m1.z-m3.z])  # c1-c3
    c32 = ti.Vector([m2.x-m3.x, m2.y-m3.y, m2.z-m3.z])  # c2-c3
    cm3 = ti.Vector([m3.x-m.x, m3.y-m.y, m3.z-m.z])  # c3-cm

    R1 = m1.w-m3.w
    R2 = m2.w-m3.w
    A = ts.dot(c31, c31)
    B = 2.0 * ts.dot(c31, c32)
    C = ts.dot(c32, c32)
    D = 2.0 * ts.dot(c31, cm3)
    E = 2.0 * ts.dot(c32, cm3)
    F = ts.dot(cm3, cm3)

    t1, t2 = -1.0, -1.0
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
        _, t11, t12 = solve_quadratic(W1, W2, W3)
        t21, t22 = H2 * t11 + K2, H2 * t12 + K2
        dis = surface_distane(m, bary_lerp(m1, m2, m3, t11, t21))
        t1, t2 = t11, t21
        if surface_distane(m, bary_lerp(m1, m2, m3, t12, t22)) < dis:
            t1, t2 = t12, t22
    elif R1 == 0.0 and R2 != 0.0:
        H1 = -B / (2.0 * A)
        K1 = -D / (2.0 * A)
        W1 = pow(2.0 * C + B * H1, 2.0) - 4.0 * R2 * \
            R2 * (C + B * H1 + A * H1 * H1)
        W2 = 2.0 * (2.0 * C + B * H1) * (B * K1 + E) - 4.0 * R2 * \
            R2 * (B * K1 + 2.0 * A * H1 * K1 + E + D * H1)
        W3 = pow(B * K1 + E, 2.0) - 4.0 * R2 * \
            R2 * (A * K1 * K1 + D * K1 + F)
        _, t21, t22 = solve_quadratic(W1, W2, W3)
        t11, t12 = H1 * t21 + K1, H1 * t22 + K1
        dis = surface_distane(m, bary_lerp(m1, m2, m3, t11, t21))
        t1, t2 = t11, t21
        if surface_distane(m, bary_lerp(m1, m2, m3, t12, t22)) < dis:
            t1, t2 = t12, t22
    else:
        L1 = 2.0 * A * R2 - B * R1
        L2 = 2.0 * C * R1 - B * R2
        L3 = E * R1 - D * R2
        if L1 == 0.0 and L2 != 0.0:
            t2 = -L3 / L2
            W1 = 4.0 * A * A - 4.0 * R1 * R1 * A
            W2 = 4.0 * A * (B * t2 + D) - 4.0 * R1 * R1 * (B * t2 + D)
            W3 = pow(B * t2 + D, 2.0) - (C * t2 * t2 + E * t2 + F)
            _, t11, t12 = solve_quadratic(W1, W2, W3)
            dis = surface_distane(m, bary_lerp(m1, m2, m3, t11, t2))
            t1 = t11
            if surface_distane(m, bary_lerp(m1, m2, m3, t12, t2)) < dis:
                t1 = t12
        elif L1 != 0.0 and L2 == 0.0:
            t1 = L3 / L1
            W1 = 4.0 * C * C - 4.0 * R2 * R2 * C
            W2 = 4.0 * C * (B * t1 + E) - 4.0 * R2 * R2 * (B * t1 + E)
            W3 = pow(B * t1 + E, 2.0) - (A * t1 * t1 + D * t1 + F)
            _, t21, t22 = solve_quadratic(W1, W2, W3)
            dis = surface_distane(m, bary_lerp(m1, m2, m3, t1, t21))
            t2 = t21
            if surface_distane(m, bary_lerp(m1, m2, m3, t1, t22)) < dis:
                t2 = t22
        else:
            H3 = L2 / L1
            K3 = L3 / L1
            W1 = (2.0 * C + B * H3)**2.0 - 4.0 * R2 * \
                R2 * (A * H3 * H3 + B * H3 + C)
            W2 = 2.0 * (2.0 * C + B * H3) * (B * K3 + E) - 4.0 * R2 * \
                R2 * (2.0 * A * H3 * K3 + B * K3 + D * H3 + E)
            W3 = (B * K3 + E)**2.0 - 4.0 * R2 * \
                R2 * (A * K3 * K3 + D * K3 + F)
            _, t21, t22 = solve_quadratic(W1, W2, W3)
            t11, t12 = H3 * t21 + K3, H3 * t22 + K3
            dis = surface_distane(m, bary_lerp(m1, m2, m3, t11, t21))
            t1, t2 = t11, t21
            if surface_distane(m, bary_lerp(m1, m2, m3, t12, t22)) < dis:
                t1, t2 = t12, t22

    if (t1+t2) < 1.0 and 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0:  # nearest sphere is inside triangle
        pass
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
    r11, r12, r21, r22 = m11.w, m12.w, m21.w, m22.w

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

    t1, t2 = -1.0, -1.0
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


##################################
## Check Intersection Algorithm ##
##################################
@ti.func
def value_of_quadratic_surface_2D(x, y, A, B, C, D, E, F):
    return A*x*x + B*x*y + C*y*y + D*x + E*y + F


@ti.func
def value_of_quadratic_surface_3D(x,  y,  z,  A1,  A2,  A3,  A4,  A5,  A6,  A7,  A8,  A9,  A10):
    return A1 * x*x + A2 * y*y + A3 * z*z + A4 * x*y + A5 * x*z + A6 * y*z + A7 * x + A8 * y + A9 * z + A10


@ti.func
def detect_cone_cone_param(A, B, C, D, E, F, space_triangle):
    # Ax^2 + By^2 + Cxy + Dx + Ey + F = 0
    min_dis = 1.0
    # case 1 x = 0, y = 0
    if (D >= 0.0 and E >= 0.0):
        f = value_of_quadratic_surface_2D(0.0, 0.0, A, B, C, D, E, F)
        min_dis = f if f < min_dis else min_dis
    # case 2 x=0, y != 0,1
    E2C = -1.0 * (E / (2.0 * C))
    if (E2C > 0.0 and E2C < 1.0):
        DB = B * E2C + D
        if (DB >= 0.0):
            f = value_of_quadratic_surface_2D(0.0, E2C, A, B, C, D, E, F)
            min_dis = f if f < min_dis else min_dis
    # case 3 x= 0, y=1
    if (B+D) >= 0.0 and (2.0*C+E) <= 0.0:
        f = value_of_quadratic_surface_2D(0.0, 1.0, A, B, C, D, E, F)
        min_dis = f if f < min_dis else min_dis
    # case 4 x!=0,1, y=0
    if A != 0.0:
        D2A = -D / (2.0 * A)
        if 0.0 < D2A < 1.0:
            EB = B * D2A + E
            if EB >= 0.0:
                f = value_of_quadratic_surface_2D(D2A, 0.0, A, B, C, D, E, F)
                min_dis = f if f < min_dis else min_dis
    # case 5 x!=0,1 y != 0,1
    delta = 4.0 * A * C - B * B
    if delta != 0.0:
        x = (B * E - 2.0 * C * D) / delta
        y = (B * D - 2.0 * A * E) / delta
        if 0.0 < x < 1.0 and 0.0 < y < 1.0:
            if not space_triangle:
                f = value_of_quadratic_surface_2D(x, y, A, B, C, D, E, F)
                min_dis = f if f < min_dis else min_dis
            else:
                if 0.0 < x+y <= 1.0:
                    f = value_of_quadratic_surface_2D(x, y, A, B, C, D, E, F)
                    min_dis = f if f < min_dis else min_dis

    # case 7 x=1 y=0
    if -(2 * A + D) >= 0.0 and B+E >= 0.0:
        f = value_of_quadratic_surface_2D(1.0, 0.0, A, B, C, D, E, F)
        min_dis = f if f < min_dis else min_dis

    if not space_triangle:
        # case 6 x != 0,1 y =1
        if A != 0.0:
            x = -(B+D) / (2.0 * A)
            CBE = 2.0 * C - (B*B + B * D) / (2.0 * A) + E
            if 0.0 < x < 1.0 and CBE <= 0.0:
                f = value_of_quadratic_surface_2D(x, 1.0, A, B, C, D, E, F)
                min_dis = f if f < min_dis else min_dis
        # case 8 x=1 y!=0,1
        y = -(B+E) / (2.0 * C)
        ABD = 2.0 * A - (B*B+B*E) / (2.0 * C) + D
        if 0.0 < y < 1.0 and ABD <= 0.0:
            f = value_of_quadratic_surface_2D(1.0, y, A, B, C, D, E, F)
            min_dis = f if f < min_dis else min_dis
        # case 9 x=1,y=1
        ABD = -(2.0 * A + B + D)
        CBE = -(2.0 * C + B + E)
        if ABD >= 0.0 and CBE >= 0.0:
            f = value_of_quadratic_surface_2D(1.0, 1.0, A, B, C, D, E, F)
            min_dis = f if f < min_dis else min_dis

    return min_dis <= 0.0


@ti.func
def detect_cone_slab_param(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10):
    # z = 0, x,y in [0,1]
    intersected = detect_cone_cone_param(A1, A4, A2, A7, A8, A10, False)
    # y = 0, x,z in [0,1]
    if not intersected:
        intersected = detect_cone_cone_param(A1, A5, A3, A7, A9, A10, False)
    # x = 0, y+z <= 1, y>=0, z>=0
    if not intersected:
        intersected = detect_cone_cone_param(A2, A6, A3, A8, A9, A10, True)
    # x = 1; y+z <=1, y>=0,z>=0
    if not intersected:
        intersected = detect_cone_cone_param(
            A2, A6, A3, A4 + A8, A5 + A9, A1 + A7 + A10, True)
    # x in [0,1]; y+z = 1, y>=0,z>=0
    if not intersected:
        intersected = detect_cone_cone_param(
            A1, A4 - A5, A2 + A3 - A6, A5 + A7, A6 + A8 - A9 - 2.0*A3, A3 + A9 + A10, False)

    # x in [0,1]; y+z <= 1, y,z >=0
    if not intersected:
        mat = ti.Matrix([
            [2.0 * A1, A4, A5],
            [A4, 2.0 * A2, A6],
            [A5, A6, 2.0 * A3]
        ])
        b = ti.Vector([-A7, -A8, -A9])
        solve = mat.inverse() @ b
        if 0.0 <= solve.x <= 1.0 and 0.0 <= solve.y <= 1.0 and 0.0 <= solve.z <= 1.0 and (0.0 <= (solve.y + solve.z) <= 1.0):
            intersected = (value_of_quadratic_surface_3D(
                solve.x, solve.y, solve.z, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) <= 0.0)

    return intersected


@ti.func
def detect_cone_cone(m11: ti.template(), m12: ti.template(), m21: ti.template(), m22: ti.template()):
    r11, r12, r21, r22 = m11.w, m12.w, m21.w, m22.w

    c12c11 = ti.Vector([m11.x-m12.x, m11.y-m12.y, m11.z-m12.z])
    c22c21 = ti.Vector([m21.x-m22.x, m21.y-m22.y, m21.z-m22.z])
    c22c12 = ti.Vector([m12.x-m22.x, m12.y-m22.y, m12.z-m22.z])

    R1 = r11 - r12
    R2 = r21 - r22
    R3 = r12 + r22

    A = ts.dot(c12c11, c12c11) - R1*R1
    B = -2.0 * ts.dot(c12c11, c22c21) - 2.0*R1*R2
    C = ts.dot(c22c21, c22c21) - R2*R2
    D = 2.0 * ts.dot(c12c11, c22c12) - 2.0*R1*R3
    E = -2.0 * ts.dot(c22c21, c22c12) - 2.0*R2*R3
    F = ts.dot(c22c12, c22c12) - R3*R3
    return detect_cone_cone_param(A, B, C, D, E, F, False)


@ti.func
def detect_cone_slab(m11: ti.template(), m12: ti.template(), m21: ti.template(), m22: ti.template(), m23: ti.template()):
    r11, r12, r21, r22, r23 = m11.w, m12.w, m21.w, m22.w, m23.w
    c12c11 = ti.Vector([m11.x-m12.x, m11.y-m12.y, m11.z-m12.z])
    c23c21 = ti.Vector([m21.x-m23.x, m21.y-m23.y, m21.z-m23.z])
    c23c22 = ti.Vector([m22.x-m23.x, m22.y-m23.y, m22.z-m23.z])
    c23c12 = ti.Vector([m12.x-m23.x, m12.y-m23.y, m12.z-m23.z])

    A1 = ts.dot(c12c11, c12c11) - (r11 - r12)*(r11 - r12)
    A2 = ts.dot(c23c21, c23c21) - (r21 - r23)*(r21 - r23)
    A3 = ts.dot(c23c22, c23c22) - (r22 - r23)*(r22 - r23)
    A4 = -2.0*(ts.dot(c12c11, c23c21) + (r11 - r12)*(r21 - r23))
    A5 = -2.0*(ts.dot(c12c11, c23c22) + (r11 - r12)*(r22 - r23))
    A6 = 2.0*(ts.dot(c23c21, c23c22) - (r21 - r23)*(r22 - r23))
    A7 = 2.0*(ts.dot(c23c12, c12c11) - (r11 - r12)*(r12 + r23))
    A8 = -2.0*(ts.dot(c23c12, c23c21) + (r21 - r23)*(r12 + r23))
    A9 = -2.0*(ts.dot(c23c12, c23c22) + (r22 - r23)*(r12 + r23))
    A10 = ts.dot(c23c12, c23c12) - (r12 + r23)*(r12 + r23)

    return detect_cone_slab_param(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10)


#################################
## Finding First-TOI Algorithm ##
#################################
@ti.func
def advance_medial_sphere(m: ti.template(), v: ti.template(), t: ti.f32):
    return ti.Vector([m.x + v.x*t, m.y + v.y*t, m.z + v.z*t, m.w])


@ti.func
def find_cloeset_t(P1, P2, P3, Sr):
    # S = (P1 t^2 + P2 t + P3)^{1/2} - Sr = 0
    C = P3 - Sr*Sr
    t = 1.0
    # if (P1 == 0.0 and C == 0.0): # if p1 = 0 p2 should = 0
    #     t = 1.0 # keeps contact, so return 1
    # else:
    # P1 != 0
    delta = P2 * P2 - 4.0 * P1 * C
    if delta <= 0.0:
        symmetric = -P2 / (2.0 * P1)
        if 0.0 <= symmetric <= 1.0:
            t = symmetric
        elif symmetric < 0.0:
            t = 0.0
        # else t = 1.0
    else:
        t = ts.clamp((-P2 - ti.sqrt(delta)) / (2.0 * P1), 0.0, 1.0)
    return t, delta


@ti.func
def get_cone_cone_toi(m11: ti.template(), m12: ti.template(), m21: ti.template(), m22: ti.template(), v11: ti.template(), v12: ti.template(), v21: ti.template(), v22: ti.template()):
    # c11 = ti.Vector([m11.x, m11.y, m11.z])
    # c12 = ti.Vector([m12.x, m12.y, m12.z])
    # c21 = ti.Vector([m21.x, m21.y, m21.z])
    # c22 = ti.Vector([m22.x, m22.y, m22.z])
    # r11, r12, r21, r22 = m11.w, m12.w, m21.w, m22.w

    c12c11 = ti.Vector([m11.x-m12.x, m11.y-m12.y, m11.z-m12.z])  # c11-c12
    c22c21 = ti.Vector([m21.x-m22.x, m21.y-m22.y, m21.z-m22.z])  # c21-c22
    c22c12 = ti.Vector([m12.x-m22.x, m12.y-m22.y, m12.z-m22.z])  # c12-c22
    v12v11 = ti.Vector([v11.x-v12.x, v11.y-v12.y, v11.z-v12.z])  # v11-v12
    v22v21 = ti.Vector([v21.x-v22.x, v21.y-v22.y, v21.z-v22.z])  # v21-v22
    v22v12 = ti.Vector([v12.x-v22.x, v12.y-v22.y, v12.z-v22.z])  # v12-v22

    # t is time, x,y are the linear interpolation parameters
    # ci = c12+v12*t + (c11-c12+v11*t-v12*t) * x = c12+v12*t + (c12c11+v12v11*t) * x
    # cj = c22+v22*t + (c21-c22+v21*t-v22*t) * y = c22+v22*t + (c22c21+v22v21*t) * y
    # Sc^2 = ||ci-cj||^2
    # = (A1t^2+A2t+A3)x^2 + (B1t^2+B2t+B3)y^2 + Ct^2 + (D1t^2+D2t+D3)xy + (E1t+E2)xt + (F1t+F2)yt + (G1t+G2)x + (H1t+H2)y + It + J
    # = (A1t^2+A2t+A3)x^2 + (B1t^2+B2t+B3)y^2 + (D1t^2+D2t+D3)xy + [E1t^2+(E2+G1)t+G2]x + [F1t^2+(F2+H1)t+H2]y + Ct^2 + It + J
    # = (A1t^2+A2t+A3)x^2 + (B1t^2+B2t+B3)y^2 + (D1t^2+D2t+D3)xy + [E1t^2+E2t+E3]x + [F1t^2+F2t+F3]y + Ct^2 + It + J

    # x^2
    A1 = ts.sqrLength(v12v11)
    A2 = 2.0 * ts.dot(c12c11, v12v11)
    A3 = ts.sqrLength(c12c11)
    # y^2
    B1 = ts.sqrLength(v22v21)
    B2 = 2.0 * ts.dot(c22c21, v22v21)
    B3 = ts.sqrLength(c22c21)
    # xy
    D1 = -2.0 * ts.dot(v12v11, v22v21)
    D2 = -2.0 * (ts.dot(v12v11, c22c21) + ts.dot(v22v21, c12c11))
    D3 = -2.0 * ts.dot(c12c11, c22c21)
    # x
    E1 = 2.0 * ts.dot(v12v11, v22v12)
    E2 = 2.0 * ts.dot(c12c11, v22v12) + 2.0 * ts.dot(c22c12, v12v11)
    E3 = 2.0 * ts.dot(c12c11, c22c12)
    # y
    F1 = -2.0 * ts.dot(v22v21, v22v12)
    F2 = -2.0 * ts.dot(c22c21, v22v12) - 2.0 * ts.dot(c22c12, v22v21)
    F3 = -2.0 * ts.dot(c22c21, c22c12)
    # t
    C = ts.sqrLength(v22v12)
    I = 2.0 * ts.dot(v22v12, c22c12)
    J = ts.sqrLength(c22c12)

    # Sr = (r11-r12)*x + (r21-r22)*y + r12+r22
    #    = R1 * x + R2 * y + R3
    R1, R2, R3 = m11.w-m12.w, m21.w-m22.w, m12.w+m22.w

    # time interval [t0,t1] (mapping to [0,1])
    t = 1.0  # start with initial value as t1
    x, y = -1.0, -1.0
    iter_num = 0
    while iter_num < 8:  # iteration number is static
        m11t = advance_medial_sphere(m11, v11, t)
        m12t = advance_medial_sphere(m12, v12, t)
        m21t = advance_medial_sphere(m21, v21, t)
        m22t = advance_medial_sphere(m22, v22, t)
        # finding neareset sphere at t
        x, y = get_cone_cone_nearest(m11t, m12t, m21t, m22t)

        # P1 = ts.sqrLength(v1t-v2t)
        # P2 = 2.0 * ts.dot(c1c2, v1t-v2t)
        # P3 = ts.sqrLength(c1c2)
        # Sr = m1t.w + m2t.w
        # finding cloesest moment of sphere x,y
        # squared distance between 2 spheres
        # S = Sc^(1/2) - Sr = (P1 t^2 + P2 t + P3)^(1/2) - Sr, S=0 means contact
        x2, y2, xy = x*x, y*y, x*y
        P1 = A1*x2 + B1*y2 + D1*xy + E1*x + F1*y+C
        P2 = A2 * x2 + B2 * y2 + D2 * xy + E2 * x + F2 * y + I
        P3 = D3 * xy + E3 * x + F3 * y + J + A3 * x2 + B3 * y2
        Sr = R1 * x + R2 * y + R3

        t_local, delta = find_cloeset_t(P1, P2, P3, Sr)
        if abs(t - t_local) < 1e-5:
            if delta <= 0.0:
                t = 1.0
            break

        # dis = surface_distane(linear_lerp(m11t, m12t, x),
        #                       linear_lerp(m21t, m22t, y))
        # print("dis:{}, x:{}, y:{}, t_local:{}".format(dis, x, y, t_local))
        # print("P1:{},P2:{},P3:{},Sr:{}".format(P1, P2, P3, Sr))

        t = t_local  # next iteration
        iter_num += 1

    # print("Iteration Num:{}".format(iter_num))
    if t == 0.0:
        t = 1.0
    return t, x, y


@ti.func
def get_sphere_slab_toi(m: ti.template(), m1: ti.template(), m2: ti.template(), m3: ti.template(), v: ti.template(), v1: ti.template(), v2: ti.template(), v3: ti.template()):
    c3c2 = ti.Vector([m2.x - m3.x, m2.y - m3.y, m2.z - m3.z])
    c3c1 = ti.Vector([m1.x - m3.x, m1.y - m3.y, m1.z - m3.z])
    cc3 = ti.Vector([m3.x - m.x, m3.y - m.y, m3.z - m.z])
    vv3 = v3 - v
    v3v2 = v2 - v3
    v3v1 = v1 - v3

    # ci = c+vt, cj = [(v1-v3)t + c1-c3] x + [(v2-v3)t + c2-c3] y + c3+v3t
    # Sc = ||cj - ci||^2
    # = (A1t^2+A2t+A3)x^2 + (B1t^2+B2t+B3)y^2 + (D1t^2+D2t+D3)xy + [E1t^2+(E2+G1)t+G2]x + [F1t^2+(F2+H1)t+H2]y + Ct^2 + It + J
    # x^2
    A1 = ts.sqrLength(v3v1)
    A2 = 2.0 * ts.dot(c3c1, v3v1)
    A3 = ts.sqrLength(c3c1)
    # y^2
    B1 = ts.sqrLength(v3v2)
    B2 = 2.0 * ts.dot(c3c2, v3v2)
    B3 = ts.sqrLength(c3c2)
    # xy
    D1 = 2.0 * ts.dot(v3v2, v3v1)
    D2 = 2.0 * (ts.dot(v3v2, c3c1) + ts.dot(v3v1, c3c2))
    D3 = 2.0 * ts.dot(c3c2, c3c1)
    # x
    E1 = 2.0 * ts.dot(v3v1, vv3)
    E2 = 2.0 * ts.dot(c3c1, vv3) + 2.0 * ts.dot(v3v1, cc3)
    E3 = 2.0 * ts.dot(c3c1, cc3)
    # y
    F1 = 2.0 * ts.dot(v3v2, vv3)
    F2 = 2.0 * ts.dot(c3c2, vv3) + 2.0 * ts.dot(v3v2, cc3)
    F3 = 2.0 * ts.dot(c3c2, cc3)
    # t^2 / t
    C = ts.sqrLength(vv3)
    I = 2.0 * ts.dot(vv3, cc3)
    J = ts.sqrLength(cc3)

    # Sr = r3 + r + (r1-r3) x + (r2-r3) y
    R1, R2, R3 = m1.w - m3.w, m2.w-m3.w, m3.w + m.w

    t = 1.0
    x, y = -1.0, -1.0
    iter_num = 0
    while iter_num < 8:
        mt = advance_medial_sphere(m, v, t)
        m1t = advance_medial_sphere(m1, v1, t)
        m2t = advance_medial_sphere(m2, v2, t)
        m3t = advance_medial_sphere(m3, v3, t)
        # finding neareset sphere at t
        x, y = get_sphere_slab_nearest(mt, m1t, m2t, m3t)

        x2, y2, xy = x*x, y*y, x*y
        P1 = A1 * x2 + B1 * y2 + D1 * xy + E1 * x + F1 * y + C
        P2 = A2 * x2 + B2 * y2 + D2 * xy + E2 * x + F2 * y + I
        P3 = A3 * x2 + B3 * y2 + D3 * xy + E3 * x + F3 * y + J
        Sr = R1 * x + R2 * y + R3

        t_local, delta = find_cloeset_t(P1, P2, P3, Sr)

        # print(bary_lerp(m1t, m2t, m3t, x, y))
        # dis = surface_distane(mt, bary_lerp(m1t, m2t, m3t, x, y))
        # print("dis:{}, x:{}, y:{}, t_local:{}".format(dis, x, y, t_local))
        # print("P1:{},P2:{},P3:{},Sr:{} \n".format(P1, P2, P3, Sr))
        if abs(t - t_local) < 1e-5:
            if delta <= 0.0:
                t = 1.0
            break

        t = t_local  # next iteration
        iter_num += 1

    # print("Iterations:{}".format(iter_num))
    if t == 0.0:
        t = 1.0
    return t, x, y


@ti.func
def get_cone_slab_toi(m11: ti.template(), m12: ti.template(), m21: ti.template(), m22: ti.template(), m23: ti.template(), v11: ti.template(), v12: ti.template(), v21: ti.template(), v22: ti.template(), v23: ti.template()):
    c11c12 = ti.Vector([m11.x-m12.x, m11.y-m12.y, m11.z-m12.z])
    c21c23 = ti.Vector([m21.x-m23.x, m21.y-m23.y, m21.z-m23.z])
    c22c23 = ti.Vector([m22.x-m23.x, m22.y-m23.y, m22.z-m23.z])
    c12c23 = ti.Vector([m12.x-m23.x, m12.y-m23.y, m12.z-m23.z])

    v11v12 = v11-v12
    v21v23 = v21-v23
    v22v23 = v22-v23
    v12v23 = v12-v23

    # x^2
    A1 = ts.sqrLength(v11v12)
    A2 = 2.0 * ts.dot(v11v12, c11c12)
    A3 = ts.sqrLength(c11c12)
    # y^2
    B1 = ts.sqrLength(v21v23)
    B2 = 2.0 * ts.dot(v21v23, c21c23)
    B3 = ts.sqrLength(c21c23)
    # z^2
    C1 = ts.sqrLength(v22v23)
    C2 = 2.0 * ts.dot(v22v23, c22c23)
    C3 = ts.sqrLength(c22c23)
    # xy
    D1 = -2.0 * ts.dot(v11v12, v21v23)
    D2 = -2.0 * (ts.dot(v11v12, c21c23) + ts.dot(c11c12, v21v23))
    D3 = -2.0 * ts.dot(c11c12, c21c23)
    # xz
    E1 = -2.0 * ts.dot(v11v12, v22v23)
    E2 = -2.0 * (ts.dot(v11v12, c22c23) + ts.dot(c11c12, v22v23))
    E3 = -2.0 * ts.dot(c11c12, c22c23)
    # yz
    F1 = 2.0 * ts.dot(v21v23, v22v23)
    F2 = 2.0 * (ts.dot(v21v23, c22c23) + ts.dot(c21c23, v22v23))
    F3 = 2.0 * ts.dot(c21c23, c22c23)
    # x
    G1 = 2.0 * ts.dot(v11v12, v12v23)
    G2 = 2.0 * (ts.dot(v11v12, c12c23) + ts.dot(c11c12, v12v23))
    G3 = 2.0 * ts.dot(c11c12, c12c23)
    # y
    H1 = -2.0 * ts.dot(v21v23, v12v23)
    H2 = -2.0 * (ts.dot(v21v23, c12c23) + ts.dot(c21c23, v12v23))
    H3 = -2.0 * ts.dot(c21c23, c12c23)
    # z
    I1 = -2.0 * ts.dot(v22v23, v12v23)
    I2 = -2.0 * (ts.dot(v22v23, c12c23) + ts.dot(v12v23, c22c23))
    I3 = -2.0 * ts.dot(c22c23, c12c23)
    # t
    J1, J2, J3 = ts.sqrLength(v12v23), 2.0 * \
        ts.dot(v12v23, c12c23), ts.sqrLength(c12c23)

    R1, R2, R3, R4 = m11.w-m12.w, m21.w-m23.w, m22.w-m23.w, m12.w+m23.w

    t = 1.0
    x, y, z = -1.0, -1.0, -1.0
    # return t, x, y, z
    iter_num = 0
    while iter_num < 8:
        m11t = advance_medial_sphere(m11, v11, t)
        m12t = advance_medial_sphere(m12, v12, t)
        m21t = advance_medial_sphere(m21, v21, t)
        m22t = advance_medial_sphere(m22, v22, t)
        m23t = advance_medial_sphere(m23, v23, t)

        # m1t = linear_lerp(m11t, m12t, x)
        # m2t = bary_lerp(m21t, m22t, m23t, y, z)
        # print(m1t)
        # print(m2t)
        # print("Last iter dis:{}".format(surface_distane(m1t, m2t)))

        x, y, z = get_cone_slab_nearest(m11t, m12t, m21t, m22t, m23t)

        x2, y2, z2, xy, xz, yz = x*x, y*y, z*z, x*y, x*z, y*z
        P1 = A1*x2+B1*y2+C1*z2+D1*xy+E1*xz+F1*yz+G1*x+H1*y+I1*z+J1
        P2 = A2*x2+B2*y2+C2*z2+D2*xy+E2*xz+F2*yz+G2*x+H2*y+I2*z+J2
        P3 = A3*x2+B3*y2+C3*z2+D3*xy+E3*xz+F3*yz+G3*x+H3*y+I3*z+J3
        Sr = R1*x+R2*y+R3*z+R4

        t_local, delta = find_cloeset_t(P1, P2, P3, Sr)

        dis = surface_distane(linear_lerp(m11t, m12t, x),
                              bary_lerp(m21t, m22t, m23t, y, z))
        # print("dis:{}, x:{}, y:{}, z:{}, t_local:{}".format(dis, x, y, z, t_local))
        # print("P1:{},P2:{},P3:{},Sr:{} \n".format(P1, P2, P3, Sr))
        if abs(t - t_local) < 1e-5:
            if delta <= 0.0:
                t = 1.0
            break

        t = t_local  # next iteration
        iter_num += 1

    # print("Iterations:{}".format(iter_num))
    if t == 0.0:
        t = 1.0
    return t, x, y, z


@ti.func
def get_cone_slab_toi_simple(m11: ti.template(), m12: ti.template(), m21: ti.template(), m22: ti.template(), m23: ti.template(), v11: ti.template(), v12: ti.template(), v21: ti.template(), v22: ti.template(), v23: ti.template()):
    t, x, y, z = 0.0, 1.0, -1.0, -1.0
    t, y, z = get_sphere_slab_toi(m11, m21, m22, m23, v11, v21, v22, v23)

    _t, a, b = get_sphere_slab_toi(m12, m21, m22, m23, v12, v21, v22, v23)
    if _t < t:
        t, x, y, z = _t, 0.0, a, b

    _t, a, b = get_cone_cone_toi(m11, m12, m21, m22, v11, v12, v21, v22)
    if _t < t:
        t, x, y, z = _t, a, b, 1.0-b

    _t, a, b = get_cone_cone_toi(m11, m12, m21, m23, v11, v12, v21, v23)
    if _t < t:
        t, x, y, z = _t, a, b, 0.0

    _t, a, b = get_cone_cone_toi(m11, m12, m22, m23, v11, v12, v22, v23)
    if _t < t:
        t, x, y, z = _t, a, 0.0, b

    return t, x, y, z


#####################
## Unit Test Class ##
#####################
@ti.data_oriented
class UnitTest:
    def __init__(self):
        self.t11 = ti.field(ti.f32, shape=())
        self.t12 = ti.field(ti.f32, shape=())
        self.t21 = ti.field(ti.f32, shape=())
        self.t22 = ti.field(ti.f32, shape=())
        self.min_dis = ti.field(ti.f32, shape=())

    @ti.kernel
    def unit_detect_cone_cone(self, m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr(), m4: ti.any_arr()) -> ti.int32:
        ti_m1 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        ti_m2 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        ti_m3 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        ti_m4 = ti.Vector([m4[0], m4[1], m4[2], m4[3]])
        intersected = detect_cone_cone(ti_m1, ti_m2, ti_m3, ti_m4)
        return ti.cast(intersected, ti.int32)

    @ti.kernel
    def unit_detect_cone_slab(self,  m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr(), m4: ti.any_arr(), m5: ti.any_arr()) -> ti.int32:
        ti_m1 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        ti_m2 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        ti_m3 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        ti_m4 = ti.Vector([m4[0], m4[1], m4[2], m4[3]])
        ti_m5 = ti.Vector([m5[0], m5[1], m5[2], m5[3]])
        intersected = detect_cone_slab(ti_m1, ti_m2, ti_m3, ti_m4, ti_m5)
        return ti.cast(intersected, ti.int32)

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
    def proof_sphere_cone(self, m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr(), steps: ti.i32):
        ti_m1 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        ti_m2 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        ti_m3 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        self.min_dis[None] = 1000000.0
        self.t21[None] = 2.0
        for i in range(steps):
            t = i / ti.cast(steps-1, ti.f32)
            m = linear_lerp(ti_m2, ti_m3, t)
            dis = surface_distane(ti_m1, m)
            ti.atomic_min(self.min_dis[None], dis)

    @ti.kernel
    def proof_sphere_slab(self, m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr(), m4: ti.any_arr(), steps: ti.i32):
        ti_m1 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        ti_m2 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        ti_m3 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        ti_m4 = ti.Vector([m4[0], m4[1], m4[2], m4[3]])
        self.min_dis[None] = 1000000.0
        real_steps = min(steps, 31620)
        for i, j in ti.ndrange(real_steps, real_steps):
            t1, t2 = i / ti.cast(real_steps, ti.f32), j / \
                ti.cast(real_steps, ti.f32)
            if t1 + t2 <= 1.0:
                m = bary_lerp(ti_m2, ti_m3, ti_m4, t1, t2)
                dis = surface_distane(ti_m1, m)
                ti.atomic_min(self.min_dis[None], dis)

    @ti.kernel
    def proof_cone_cone(self, m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr(), m4: ti.any_arr(), steps: ti.i32):
        ti_m1 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        ti_m2 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        ti_m3 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        ti_m4 = ti.Vector([m4[0], m4[1], m4[2], m4[3]])
        self.min_dis[None] = 1000000.0
        real_steps = min(steps, 31620)
        for i, j in ti.ndrange(real_steps, real_steps):
            t1, t2 = i / ti.cast(real_steps-1, ti.f32), j / \
                ti.cast(real_steps-1, ti.f32)
            tm1 = linear_lerp(ti_m1, ti_m2, t1)
            tm2 = linear_lerp(ti_m3, ti_m4, t2)
            dis = surface_distane(tm1, tm2)
            ti.atomic_min(self.min_dis[None], dis)

    @ti.kernel
    def proof_cone_slab(self, m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr(), m4: ti.any_arr(), m5: ti.any_arr(), steps: ti.i32):
        ti_m1 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        ti_m2 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        ti_m3 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        ti_m4 = ti.Vector([m4[0], m4[1], m4[2], m4[3]])
        ti_m5 = ti.Vector([m5[0], m5[1], m5[2], m5[3]])
        self.min_dis[None] = 1000000.0
        real_steps = min(steps, 5000)
        for i, j in ti.ndrange(real_steps, real_steps):
            t1, t2 = i / ti.cast(real_steps-1, ti.f32), j / \
                ti.cast(real_steps-1, ti.f32)
            for k in range(real_steps):
                t3 = (1.0-t2) * k / ti.cast(real_steps-1, ti.f32)
                tm1 = linear_lerp(ti_m1, ti_m2, t1)
                tm2 = bary_lerp(ti_m3, ti_m4, ti_m5, t2, t3)
                dis = surface_distane(tm1, tm2)
                ti.atomic_min(self.min_dis[None], dis)

    @ti.kernel
    def moving_cone_cone(self, m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr(), m4: ti.any_arr(), v11: ti.any_arr(), v12: ti.any_arr(), v21: ti.any_arr(), v22: ti.any_arr()) -> ti.f32:
        m11 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        m12 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        m21 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        m22 = ti.Vector([m4[0], m4[1], m4[2], m4[3]])
        t = 0.0
        if detect_cone_cone(m11, m12, m21, m22):
            print("Should start in an intersection free state!!!")
        else:
            _v11 = ti.Vector([v11[0], v11[1], v11[2]])
            _v12 = ti.Vector([v12[0], v12[1], v12[2]])
            _v21 = ti.Vector([v21[0], v21[1], v21[2]])
            _v22 = ti.Vector([v22[0], v22[1], v22[2]])

            t, x, y = get_cone_cone_toi(
                m11, m12, m21, m22, _v11, _v12, _v21, _v22)
            m1r = linear_lerp(advance_medial_sphere(m11, _v11, t),
                              advance_medial_sphere(m12, _v12, t), x)
            m2r = linear_lerp(advance_medial_sphere(m21, _v21, t),
                              advance_medial_sphere(m22, _v22, t), y)
            print("TOI:{}, x:{}, y:{}".format(t, x, y))
            print("Surace Distance:{}".format(surface_distane(m1r, m2r)))
        return t

    @ti.kernel
    def moving_sphere_slab(self, m: ti.any_arr(), m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr(), v: ti.any_arr(), v1: ti.any_arr(), v2: ti.any_arr(), v3: ti.any_arr()) -> ti.f32:
        ti_m = ti.Vector([m[0], m[1], m[2], m[3]])
        ti_m1 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        ti_m2 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        ti_m3 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        t = 0.0
        t1, t2 = get_sphere_slab_nearest(ti_m, ti_m1, ti_m2, ti_m3)
        if surface_distane(ti_m, bary_lerp(ti_m1, ti_m2, ti_m3, t1, t2)) <= 0.0:
            print("Should start in an intersection free state!!!")
        else:
            _v = ti.Vector([v[0], v[1], v[2]])
            _v1 = ti.Vector([v1[0], v1[1], v1[2]])
            _v2 = ti.Vector([v2[0], v2[1], v2[2]])
            _v3 = ti.Vector([v3[0], v3[1], v3[2]])
            t, x, y = get_sphere_slab_toi(
                ti_m, ti_m1, ti_m2, ti_m3, _v, _v1, _v2, _v3)
            print("TOI:{}, x:{}, y:{}".format(t, x, y))
            mt = advance_medial_sphere(ti_m, _v, t)
            mst = bary_lerp(advance_medial_sphere(ti_m1, _v1, t), advance_medial_sphere(
                ti_m2, _v2, t), advance_medial_sphere(ti_m3, _v3, t), x, y)
            print("Surace Distance:{}".format(surface_distane(mt, mst)))
        return t

    @ti.kernel
    def moving_cone_slab(self, m11: ti.any_arr(), m12: ti.any_arr(), m21: ti.any_arr(), m22: ti.any_arr(), m23: ti.any_arr(), v11: ti.any_arr(), v12: ti.any_arr(), v21: ti.any_arr(), v22: ti.any_arr(), v23: ti.any_arr()) -> ti.f32:
        m11t = ti.Vector([m11[0], m11[1], m11[2], m11[3]])
        m12t = ti.Vector([m12[0], m12[1], m12[2], m12[3]])
        m21t = ti.Vector([m21[0], m21[1], m21[2], m21[3]])
        m22t = ti.Vector([m22[0], m22[1], m22[2], m22[3]])
        m23t = ti.Vector([m23[0], m23[1], m23[2], m23[3]])
        t = 0.0
        t1, t2, t3 = get_cone_slab_nearest(m11t, m12t, m21t, m22t, m23t)
        if surface_distane(linear_lerp(m11t, m12t, t1), bary_lerp(m21t, m22t, m23t, t2, t3)) <= 0.0:
            print("Should start in an intersection free state!!!")
        else:
            _v11 = ti.Vector([v11[0], v11[1], v11[2]])
            _v12 = ti.Vector([v12[0], v12[1], v12[2]])
            _v21 = ti.Vector([v21[0], v21[1], v21[2]])
            _v22 = ti.Vector([v22[0], v22[1], v22[2]])
            _v23 = ti.Vector([v23[0], v23[1], v23[2]])
            t, x, y, z = get_cone_slab_toi(
                m11t, m12t, m21t, m22t, m23t, _v11, _v12, _v21, _v22, _v23)
            print("TOI:{}, x:{}, y:{}, z:{}".format(t, x, y, z))
            mct = advance_medial_sphere(linear_lerp(
                m11t, m12t, x), linear_lerp(_v11, _v12, x), t)
            mst = advance_medial_sphere(
                bary_lerp(m21t, m22t, m23t, y, z), bary_lerp(_v21, _v22, _v23, y, z), t)
            print("Surace Distance:{}".format(surface_distane(mct, mst)))
        return t

    @ti.kernel
    def cone_cone_performance(self, m1: ti.any_arr(), m2: ti.any_arr(), m3: ti.any_arr(), m4: ti.any_arr(), v11: ti.any_arr(), v12: ti.any_arr(), v21: ti.any_arr(), v22: ti.any_arr(), steps: ti.int32):
        m11 = ti.Vector([m1[0], m1[1], m1[2], m1[3]])
        m12 = ti.Vector([m2[0], m2[1], m2[2], m2[3]])
        m21 = ti.Vector([m3[0], m3[1], m3[2], m3[3]])
        m22 = ti.Vector([m4[0], m4[1], m4[2], m4[3]])
        _v11 = ti.Vector([v11[0], v11[1], v11[2]])
        _v12 = ti.Vector([v12[0], v12[1], v12[2]])
        _v21 = ti.Vector([v21[0], v21[1], v21[2]])
        _v22 = ti.Vector([v22[0], v22[1], v22[2]])

        for i in range(steps):
            _, _, _ = get_cone_cone_toi(
                m11, m12, m21, m22, _v11, _v12, _v21, _v22)
