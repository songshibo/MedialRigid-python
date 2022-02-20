import taichi as ti
import numpy as np
import taichi_glsl as ts

ti.init(arch=ti.gpu, debug=True, kernel_profiler=True)


@ti.func
def linear_lerp(v1: ti.template(), v2: ti.template(), t: ti.f32) -> ti.template():
    return v1 * t + v2 * (1-t)


@ti.func
def bary_lerp(v1: ti.template(), v2: ti.template(), v3: ti.template(), t1: ti.f32, t2: ti.f32) -> ti.template():
    return v1 * t1 + v2 * t2 + v3 * (1.0 - t1 - t2)


@ti.func
def surface_distane(m1: ti.template(), m2: ti.template()) -> ti.f32:
    c1 = ti.Vector([m1.x, m1.y, m1.z])
    c2 = ti.Vector([m2.x, m2.y, m2.z])
    return ts.length(c1-c2) - (m1.w + m2.w)


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


cq = ti.Vector([0.2, 0.5, 0.35, 0.1])
cone_m1 = ti.Vector([0.0, 0.0, 0.0, 0.1])
cone_m2 = ti.Vector([1.0, 0.0, 0.0, 0.3])
cone_m3 = ti.Vector([0.5, 0.0, 1.0, 0.2])

s = ti.field(ti.f32, shape=())


@ti.kernel
def unit_test():
    # print(get_sphere_cone_nearest(cq, cone_m1, cone_m2))
    t1, t2 = get_sphere_slab_nearest(cq, cone_m1, cone_m2, cone_m3)
    print("t1:{},t2:{}".format(t1, t2))
    min_dis = surface_distane(cq, bary_lerp(cone_m1, cone_m2, cone_m3, t1, t2))
    print(min_dis)
    # for i, j in ti.ndrange(1000, 1000):
    #     t1, t2 = get_sphere_slab_nearest(cq, cone_m1, cone_m2, cone_m3)
    #     min_dis = surface_distane(cq, bary_lerp(
    #         cone_m1, cone_m2, cone_m3, t1, t2))
    tt1, tt2 = 0.0, 0.0
    total_num = 0
    for i, j in ti.ndrange(1000, 1000):
        tt1 += i * 1.0 / 1000
        tt2 += j * 1.0 / 1000
        dis = surface_distane(cq, bary_lerp(
            cone_m1, cone_m2, cone_m3, tt1, tt2))
        if dis > min_dis:
            # print("{},{}".format(tt1, tt2))
            tt1 += 0.0
        # ti.atomic_add(total_num, 1)
    print("failed num:{}".format(total_num))


unit_test()
ti.print_kernel_profile_info('trace')
