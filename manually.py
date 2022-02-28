import numpy as np
import taichi as ti
import taichi_glsl as ts

ti.init()


@ti.kernel
def surface_distane(c1: ti.template(), c2: ti.template(), r1: ti.f32, r2: ti.f32) -> ti.f32:
    return ts.length(c1-c2) - (r1 + r2)


c11 = ti.Vector([0.0, 0.7, 0.5])
r11 = 0.3
v11 = ti.Vector([0.0, -1, 0.0])
c12 = ti.Vector([0.0, 1.0, 0.0])
r12 = 0.45
v12 = ti.Vector([0.0, -0.5, 0.3])

c21 = ti.Vector([0.5, 0.0, 0.0])
r21 = 0.35
v21 = ti.Vector([0.0, 0.0, 0.0])
c22 = ti.Vector([-0.5, 0.0, 0.0])
r22 = 0.5
v22 = ti.Vector([0.0, 0.0, 0.0])
c23 = ti.Vector([0.0, 0.0, -0.6])
r23 = 0.2
v23 = ti.Vector([0.0, 0.0, 0.0])


c11c12 = c11-c12
c21c23 = c21-c23
c22c23 = c22-c23
c12c23 = c12-c23
v11v12 = v11-v12
v21v23 = v21-v23
v22v23 = v22-v23
v12v23 = v12-v23

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
J1 = ts.sqrLength(v12v23)
J2 = 2.0 * ts.dot(v12v23, c12c23)
J3 = ts.sqrLength(c12c23)
R1, R2, R3, R4 = r11-r12, r21-r23, r22-r23, r12+r23

x, y, z, t = 0.0, 1.0, 0.0, 0.0
cx = (c11 + v11 * t) * x + (c12 + v12 * t) * (1.0 - x)
rx = r11 * x + r12 * (1.0-x)
cyz = (c21 + v21 * t) * y + (c22 + v22 * t) * z + (c23 + v23 * t) * (1.0-y-z)
ryz = r21 * y + r22 * z + r23 * (1.0-y-z)
print(ts.sqrLength(cx-cyz))
print(rx + ryz)
print(surface_distane(cx, cyz, rx, ryz))

x2, y2, z2, xy, xz, yz = x*x, y*y, z*z, x*y, x*z, y*z
sc = (A1*x2+B1*y2+C1*z2+D1*xy+E1*xz+F1*yz+G1*x+H1*y+I1*z+J1) * t*t +\
     (A2*x2+B2*y2+C2*z2+D2*xy+E2*xz+F2*yz+G2*x+H2*y+I2*z+J2) * t +\
     (A3*x2+B3*y2+C3*z2+D3*xy+E3*xz+F3*yz+G3*x+H3*y+I3*z+J3)
print(sc)
print(R1*x+R2*y+R3*z+R4)
