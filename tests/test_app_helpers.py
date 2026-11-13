import os
import unittest

from app import build_material_rows, build_uv_previews
from ghg_parser import GHGModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    return GHGModel(os.path.join(ROOT, name))


class AppHelpersTest(unittest.TestCase):
    def test_material_rows(self):
        model = load("indianajones_dj_pc.ghg")
        rows = build_material_rows(model)

        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0][0], 0)
        self.assertEqual(rows[0][2], "#0 1024x1024")
        self.assertGreater(rows[0][5], 0)
        self.assertEqual(rows[0][4], 1.0)
        self.assertNotEqual(rows[0][6], "-")
        self.assertNotEqual(rows[0][7], "-")

    def test_uv_previews(self):
        model = load("biplane_pc.ghg")
        previews = build_uv_previews(model)

        self.assertEqual(len(previews), 21)
        for image, caption in previews:
            self.assertLessEqual(max(image.size), 512)
            self.assertTrue(caption.startswith("Mat "))


if __name__ == "__main__":
    unittest.main()
