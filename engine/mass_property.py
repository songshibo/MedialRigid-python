import wildmeshing as wm
import os
import numpy as np

C_canonical = np.array([[1.0 / 60.0, 1.0 / 120.0, 1.0 / 120.0],
                        [1.0 / 120.0, 1.0 / 60.0, 1.0 / 120.0],
                        [1.0 / 120.0, 1.0 / 120.0, 1.0 / 60.0]])


def tetrahedralize_from_file_with_output(filepath, overwrite=False):
    pre, _ = os.path.splitext(filepath)
    outpath = pre + ".msh"
    if (not os.path.exists(outpath)) or overwrite:
        wm.tetrahedralize(filepath, outpath)
    # return saved msh file path
    return outpath


def compute_mass_center_with_tet(VT, TT, vol, total_vol):
    tmp_center = np.zeros(3)
    for idx, t in enumerate(TT):
        v0 = VT[t[0], :]
        v1 = VT[t[1], :]
        v2 = VT[t[2], :]
        v3 = VT[t[3], :]
        c = 0.25 * (v0 + v1 + v2 + v3)
        tmp_center += c * vol[idx]

    return tmp_center / total_vol


def compute_mass_center_with_triangle(v, f):
    # return np.mean(v, axis=0)
    total_m = 0
    tmp_center = np.zeros(3)
    for tri in f:
        v1 = v[tri[0], :]
        v2 = v[tri[1], :]
        v3 = v[tri[2], :]
        c = (v1+v2+v3) / 3.0
        area = np.linalg.norm(
            np.cross(v1, v2) + np.cross(v2, v3) + np.cross(v3, v1)) * 0.5

        total_m += area
        tmp_center += c * area

    return tmp_center / total_m


def affine_inertia(v1, v2, v3):
    global C_canonical
    A = np.zeros([3, 3])
    A[:, 0] = v1
    A[:, 1] = v2
    A[:, 2] = v3
    return np.linalg.det(A) * A.dot(C_canonical).dot(A.T)


# Refer to http://number-none.com/blow/inertia/index.html
def compute_inertia_tensor_with_tet(VT, TT, com):
    C_sum = np.zeros([3, 3])
    for tet in TT:
        v0 = VT[tet[0], :] - com
        v1 = VT[tet[1], :] - com
        v2 = VT[tet[2], :] - com
        v3 = VT[tet[3], :] - com
        C_sum = C_sum + affine_inertia(v1, v2, v3)
        C_sum = C_sum + affine_inertia(v1, v0, v2)
        C_sum = C_sum + affine_inertia(v1, v3, v0)
        C_sum = C_sum + affine_inertia(v0, v3, v2)

    return np.trace(C_sum) * np.eye(3) - C_sum
