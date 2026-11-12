import os
import tempfile
import unittest

import numpy as np

from ghg_parser import GHGModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    return GHGModel(os.path.join(ROOT, name))


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
                "vertices": 920,
                "faces": 1418,
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
                "draws": 44,
                "vertices": 4757,
                "faces": 7183,
                "textures": 24,
                "geometry_offset": "0x037074",
                "bounds": [[-0.4572, -0.3658, -0.8382], [1.2192, 0.3962, 0.8382]],
            },
        )

    def test_indianajones(self):
        self.check_model(
            "indianajones_dj_pc.ghg",
            {
                "vertex_buffers": 3,
                "draws": 19,
                "vertices": 4605,
                "faces": 8752,
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


if __name__ == "__main__":
    unittest.main()
