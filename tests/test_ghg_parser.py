import json
import os
import struct
import tempfile
import unittest

import numpy as np

from ghg_parser import GHGModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    return GHGModel(os.path.join(ROOT, name))


def read_glb(path):
    with open(path, "rb") as handle:
        data = handle.read()
    json_length, _ = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20:20 + json_length].decode("utf-8"))
    bin_header = 20 + json_length
    bin_length, _ = struct.unpack_from("<II", data, bin_header)
    binary = data[bin_header + 8:bin_header + 8 + bin_length]
    return gltf, binary


def read_glb_json(path):
    return read_glb(path)[0]


def read_vec2_accessor(gltf, binary, accessor_index):
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]
    return np.frombuffer(
        binary, dtype="<f4", count=count * 2, offset=offset
    ).reshape(-1, 2)


class ParserSamplesTest(unittest.TestCase):
    def check_model(self, name, expected):
        model = load(name)
        stats = model.stats

        self.assertEqual(stats["num_vertex_buffers"], expected["vertex_buffers"])
        self.assertEqual(stats["num_draws"], expected["draws"])
        self.assertEqual(stats["num_vertices"], expected["vertices"])
        self.assertEqual(stats["num_faces"], expected["faces"])
        self.assertEqual(stats["num_textures"], expected["textures"])
        self.assertEqual(stats["geometry_offset"], expected["geometry_offset"])
        self.assertTrue(np.allclose(stats["bounds"], expected["bounds"], atol=1e-3))

        self.assertTrue(model.draws)
        for draw in model.draws:
            self.assertGreater(draw["triangles"], 0)
            self.assertGreater(draw["stored_normal_dot"], 0.85)

    def test_scooter(self):
        self.check_model(
            "scooter_pc.ghg",
            {
                "vertex_buffers": 1,
                "draws": 8,
                "vertices": 924,
                "faces": 1434,
                "textures": 2,
                "geometry_offset": "0x006514",
                "bounds": [[-0.1524, -0.0007, -0.2637], [0.1524, 0.3441, 0.2858]],
            },
        )

    def test_biplane(self):
        self.check_model(
            "biplane_pc.ghg",
            {
                "vertex_buffers": 2,
                "draws": 48,
                "vertices": 4766,
                "faces": 7230,
                "textures": 24,
                "geometry_offset": "0x037074",
                "bounds": [[-0.4572, -0.3658, -0.8382], [1.2192, 0.3962, 0.8382]],
            },
        )

    def test_biplane_missing_surfaces_recovered(self):
        model = load("biplane_pc.ghg")
        self.assertEqual(len(model.draws), 48)
        self.assertEqual(len(model.skeleton_parts), 48)
        self.assertFalse(
            any("mapping partiel" in warning for warning in model.warnings)
        )

        vertices = np.asarray(model.vertices)
        faces = np.asarray(model.faces)

        def has_face(predicate):
            return any(predicate(vertices[face]) for face in faces)

        def left_float_bottom(points):
            return (
                np.all(np.abs(points[:, 1] + 0.3353) < 2e-3)
                and np.all(points[:, 2] > 0.22)
                and np.all(points[:, 2] < 0.39)
                and np.all(points[:, 0] > -0.39)
                and np.all(points[:, 0] < 0.39)
            )

        def left_float_end(points):
            return (
                np.all(np.abs(points[:, 0] + 0.381) < 2e-3)
                and np.all(points[:, 1] > -0.34)
                and np.all(points[:, 1] < -0.30)
                and np.all(points[:, 2] > 0.22)
                and np.all(points[:, 2] < 0.39)
            )

        def right_aileron_underside(points):
            return (
                np.all(np.abs(points[:, 1] - 0.1829) < 2e-3)
                and np.all(points[:, 2] < -0.07)
                and np.all(points[:, 2] > -0.39)
                and np.all(np.abs(points[:, 0]) < 0.08)
            )

        self.assertTrue(has_face(left_float_bottom))
        self.assertTrue(has_face(left_float_end))
        self.assertTrue(has_face(right_aileron_underside))

        # The cargo-box top is intentionally open in the source data: the
        # crate has a clean 18-edge rim at y=0 and no top face.

    def test_indianajones(self):
        self.check_model(
            "indianajones_dj_pc.ghg",
            {
                "vertex_buffers": 3,
                "draws": 19,
                "vertices": 4602,
                "faces": 8780,
                "textures": 4,
                "geometry_offset": "0x17C784",
                "bounds": [[-0.1187, -0.0001, -0.0887], [0.1187, 0.4056, 0.0590]],
            },
        )

    def test_gsnh_bone_mapping(self):
        scooter = load("scooter_pc.ghg")
        self.assertEqual(scooter.skeleton_bones, ["scooter", "joint1", "joint3"])
        self.assertIn("joint1", {draw["bone_name"] for draw in scooter.draws})
        self.assertIn("joint3", {draw["bone_name"] for draw in scooter.draws})

        biplane = load("biplane_pc.ghg")
        self.assertIn("Propeller", biplane.skeleton_bones)
        self.assertIn("Propeller", {draw["bone_name"] for draw in biplane.draws})

        indy = load("indianajones_dj_pc.ghg")
        self.assertIn("Head", indy.skeleton_bones)
        self.assertIn("Head", {draw["bone_name"] for draw in indy.draws})
        self.assertGreater(indy.stats["bounds"][1][1], 0.4)

    def test_material_texture_mapping(self):
        scooter = load("scooter_pc.ghg")
        self.assertEqual(len(scooter.materials), 6)
        self.assertEqual(
            [material["texture"] for material in scooter.materials],
            [-1, -1, -1, -1, -1, -1],
        )

        biplane = load("biplane_pc.ghg")
        self.assertEqual(len(biplane.materials), 21)
        self.assertEqual(
            [material["texture"] for material in biplane.materials],
            [-1, -1, 1, -1, -1, -1, -1, 4, -1, 6, 8, -1, 10, -1, 12, 13, 15, -1, -1, 17, -1],
        )
        self.assertEqual(len(biplane.skeleton_part_materials), 48)
        self.assertEqual(
            sorted(set(biplane.face_materials.tolist())),
            list(range(21)),
        )

        indy = load("indianajones_dj_pc.ghg")
        self.assertEqual(len(indy.materials), 7)
        self.assertEqual(
            [material["texture"] for material in indy.materials],
            [0, 0, 0, 0, 0, 2, 3],
        )
        self.assertEqual(sorted(set(indy.face_materials.tolist())), list(range(7)))
        self.assertEqual(sorted(set(draw["texture"] for draw in indy.draws)), [0, 2, 3])

    def test_uvs_are_valid(self):
        for name in ("scooter_pc.ghg", "biplane_pc.ghg", "indianajones_dj_pc.ghg"):
            model = load(name)
            self.assertTrue(np.isfinite(model.uvs).all(), name)
            self.assertLess(float(np.abs(model.uvs).max()), 2.0, name)

    def test_invalid_signature(self):
        with self.assertRaises(ValueError):
            GHGModel(b"not a ghg file")

    def test_glb_export(self):
        model = load("scooter_pc.ghg")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scooter.glb")
            model.export_glb(path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)

            gltf = read_glb_json(path)
            self.assertEqual(len(gltf["materials"]), 6)
            self.assertNotIn("images", gltf)
            self.assertNotIn("textures", gltf)

    def test_textured_glb_export(self):
        model = load("indianajones_dj_pc.ghg")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "indy.glb")
            model.export_glb(path)

            gltf = read_glb_json(path)
            self.assertEqual(len(gltf["materials"]), 7)
            expected_images = len(
                {material["texture"] for material in model.materials if material["texture"] >= 0}
            )
            self.assertEqual(
                len({image["bufferView"] for image in gltf["images"]}),
                expected_images,
            )
            self.assertEqual(len(gltf["textures"]), len(gltf["images"]))

            textured = [
                material for material in gltf["materials"]
                if "baseColorTexture" in material.get("pbrMetallicRoughness", {})
            ]
            self.assertEqual(len(textured), 7)
            self.assertEqual(gltf["materials"][0].get("alphaMode"), "OPAQUE")
            self.assertEqual(gltf["materials"][5].get("alphaMode"), "BLEND")
            self.assertEqual(gltf["materials"][6].get("alphaMode"), "BLEND")

            for mesh in gltf["meshes"]:
                for primitive in mesh["primitives"]:
                    self.assertIn("TEXCOORD_0", primitive["attributes"])
                    self.assertIn("material", primitive)

    def test_glb_uv_orientation(self):
        model = load("indianajones_dj_pc.ghg")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "indy.glb")
            model.export_glb(path)

            gltf, binary = read_glb(path)
            mesh = next(
                mesh for mesh in gltf["meshes"] if mesh.get("name") == "material_00"
            )
            accessor = mesh["primitives"][0]["attributes"]["TEXCOORD_0"]
            exported_uvs = read_vec2_accessor(gltf, binary, accessor)

            face_indices = np.where(model._export_face_materials == 0)[0]
            used = np.unique(model._export_faces[face_indices])
            source_uvs = model._export_uvs[used]

            self.assertTrue(np.allclose(exported_uvs.min(0), source_uvs.min(0), atol=1e-5))
            self.assertTrue(np.allclose(exported_uvs.max(0), source_uvs.max(0), atol=1e-5))

    def test_biplane_textured_glb_export(self):
        model = load("biplane_pc.ghg")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "biplane.glb")
            model.export_glb(path)

            gltf = read_glb_json(path)
            self.assertEqual(len(gltf["materials"]), 21)
            expected_images = len({m["texture"] for m in model.materials if m["texture"] >= 0})
            self.assertEqual(len(gltf["images"]), expected_images)
            self.assertEqual(len(gltf["textures"]), expected_images)


if __name__ == "__main__":
    unittest.main()
