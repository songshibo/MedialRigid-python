import igl
import os
from util import *
from .mass_property import *
import time


def compute_bouding_box(v, _scale):
    verts = v
    for vv in verts:
        vv *= _scale
    min_corner = np.min(verts, axis=0)
    max_corner = np.max(verts, axis=0)
    return np.array([min_corner, max_corner])


class MeshObject:
    def __init__(self, name: str, path: str, position, rotation, scale, index, mass, polyscope):
        self.name = name
        self.index = index
        self.position = position
        self.orientation = euler_to_quaternion(rotation)
        self.scale = scale
        self.TRS = identity()
        self.compute_transform()

        VT, TT = igl.read_msh(tetrahedralize_from_file_with_output(path))
        v, f = igl.read_triangle_mesh(path)
        # self.mesh = polyscope.register_volume_mesh(self.name, VT, tets=TT)
        self.mesh = polyscope.register_surface_mesh(self.name, v, f)
        self.mesh.set_transform(self.TRS)
        self.bbox = compute_bouding_box(v, self.scale)
        print("   local AABB:\n   min: {}\n   max: {}".format(
            self.bbox[0], self.bbox[1]))

        # mass properties
        self.mass = mass
        self.mass_center = None
        self.inertia = None
        self.compute_mass_properties(VT, TT)
        # Medial Axis Transform info
        self.ma_verts = None
        self.ma_edges = None
        self.ma_faces = None
        self.load_ma(path)

    def normalize_model(self, path):
        if np.isclose(np.max(self.scale), 1.0):
            print("{} is already normalized".format(self.name))
            return
        verts, faces = igl.read_triangle_mesh(path)
        for idx, _ in enumerate(verts):
            verts[idx] *= self.scale
        self.mesh.update_vertex_positions(verts)
        # reset scale
        self.scale = np.ones(3)
        self.compute_transform()
        self.mesh.set_transform(self.TRS)
        print("Save normalized model(overwrite)")
        igl.write_triangle_mesh(path, verts, faces)

    def re_tetrahedralize(self, path):
        tetrahedralize_from_file_with_output(path, True)
        print("{} is re-tetrahedralized".format(self.name))

    def compute_mass_properties(self, VT, TT):
        start_time = time.time()
        vol = igl.volume(VT, TT)
        total_V = np.sum(np.absolute(vol))
        print("[{}]\n   original total volume: {}".format(self.name, total_V))

        if self.mass < 0.0:
            density = 1.0
            self.mass = total_V
        else:
            density = self.mass / total_V
        self.mass_center = compute_mass_center_with_tet(VT, TT, vol, total_V)
        print("   mass center: {}".format(self.mass_center))
        print("--- %.2f seconds ---" % (time.time() - start_time))
        self.inertia = compute_inertia_tensor_with_tet(
            VT, TT, self.mass_center, density)

    def load_ma(self, filepath):
        pre, _ = os.path.splitext(filepath)
        ma_path = pre + ".ma"
        if not os.path.exists(ma_path):
            print("   [No related ma file founded]".format(self.name))
            return
        file = open(ma_path, 'r')
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
