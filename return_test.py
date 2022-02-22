import taichi as ti

ti.init(arch=ti.gpu)


@ti.kernel
def valid_matrix_return() -> ti.types.matrix(2, 2, dtype=ti.f32):
    return ti.Matrix([[1.0, 0.0], [0.0, 1.0]])


@ti.kernel
def valid_return():
    return ti.Matrix([[1.0, 0.0], [0.0, 1.0]])


a = valid_matrix_return()
