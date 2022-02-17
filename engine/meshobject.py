import igl
from util import *


def compute_mass_center(v):
    return np.mean(v, axis=0)


# Refer to http://number-none.com/blow/inertia/index.html
def compute_inertia_const_density(v, f, center):
    C_canonical = np.array([[1.0 / 60.0, 1.0 / 120.0, 1.0 / 120.0],
                            [1.0 / 120.0, 1.0 / 60.0, 1.0 / 120.0],
                            [1.0 / 120.0, 1.0 / 120.0, 1.0 / 60.0]])

    C_sum = np.zeros([3, 3])
    for tri in f:
        # compute Covariance of each tetrahedron
        v1 = v[tri[0], :]
        v2 = v[tri[1], :]
        v3 = v[tri[2], :]
        A = np.zeros([3, 3])
        A[:, 0] = v1 - center
        A[:, 1] = v2 - center
        A[:, 2] = v3 - center
        C = np.linalg.det(A) * A.dot(C_canonical).dot(A.T)
        C_sum = C_sum + C
    return np.trace(C_sum) * np.eye(3) - C_sum


class MeshObject:
    def __init__(self, name: str, path: str, position, rotation, index, polyscope):
        self.name = name
        self.index = index
        v, f = igl.read_triangle_mesh(path)
        self.mesh = polyscope.register_surface_mesh(self.name, v, f)
        self.position = position
        self.orientation = euler_to_quaternion(rotation)
        print(self.orientation)
        self.scale = self.mesh.get_transform().diagonal()[:3]
        self.TRS = identity()
        self.compute_transform()
        self.mesh.set_transform(self.TRS)

        self.mass_center = compute_mass_center(v)
        self.inertia = compute_inertia_const_density(v, f, self.mass_center)

        # Medial Axis Transform info
        self.ma_verts = None
        self.ma_edges = None
        self.ma_faces = None

    def load_ma(self, filepath):
        file = open(filepath, 'r')
        first_line = file.readline().rstrip()
        vcount, ecount, fcount = [int(x) for x in first_line.split()]
        assert vcount != 0, "No Medial Vertices!"
        # line number
        lineno = 1

        verts, faces, edges = [], [], []

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
            edges.append(np.array([int(e[1]), int(e[2])]))
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
            faces.append(np.array([int(f[1]), int(f[2]), int(f[3])]))
            lineno += 1
            i += 1

        self.ma_verts, self.ma_edges, self.ma_faces = np.array(verts), np.array(
            edges, dtype=np.uint8), np.array(faces, dtype=np.uint8)

    def compute_transform(self):
        self.TRS = TRS(self.position, self.orientation, self.scale)

    def update_transform(self, ti_result):
        self.position = ti_result['pos'][self.index]
        self.orientation = ti_result['rot'][self.index]
        self.compute_transform()
        self.mesh.set_transform(self.TRS)
