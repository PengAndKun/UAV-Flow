"""
Layer 1 — Unit tests for house_registry.py
No server, no Unreal Engine needed.

Run:
    cd UAV-Flow-Eval
    python -m pytest test_house_registry.py -v
  or without pytest:
    python test_house_registry.py
"""
import json
import math
import os
import tempfile
import unittest

from house_registry import House, HouseRegistry, HouseStatus


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_config(tmp_dir: str, houses: list) -> str:
    path = os.path.join(tmp_dir, "houses_config.json")
    with open(path, "w") as f:
        json.dump({
            "world_bounds": {"min_x": 0, "min_y": 0, "max_x": 6000, "max_y": 4000},
            "houses": houses,
        }, f)
    return path


THREE_HOUSES = [
    {"id": "A", "name": "House A", "center_x": 1000.0, "center_y": 500.0,
     "center_z": 200.0, "approach_z": 600.0, "radius_cm": 700.0, "entry_yaw_hint": 0.0},
    {"id": "B", "name": "House B", "center_x": 3000.0, "center_y": 500.0,
     "center_z": 200.0, "approach_z": 600.0, "radius_cm": 700.0, "entry_yaw_hint": 90.0},
    {"id": "C", "name": "House C", "center_x": 2000.0, "center_y": 3000.0,
     "center_z": 200.0, "approach_z": 600.0, "radius_cm": 700.0, "entry_yaw_hint": 180.0},
]


# ──────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────

class TestHouseRegistryLoad(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = _make_config(self.tmp, THREE_HOUSES)

    def test_loads_all_houses(self):
        r = HouseRegistry(self.cfg)
        self.assertEqual(len(r.get_all_houses()), 3)

    def test_house_ids(self):
        r = HouseRegistry(self.cfg)
        ids = {h.id for h in r.get_all_houses()}
        self.assertEqual(ids, {"A", "B", "C"})

    def test_initial_status_unsearched(self):
        r = HouseRegistry(self.cfg)
        for h in r.get_all_houses():
            self.assertEqual(h.status, HouseStatus.UNSEARCHED)

    def test_missing_file_creates_empty_registry(self):
        r = HouseRegistry(os.path.join(self.tmp, "nonexistent.json"))
        self.assertEqual(len(r.get_all_houses()), 0)

    def test_get_house_by_id(self):
        r = HouseRegistry(self.cfg)
        h = r.get_house("B")
        self.assertIsNotNone(h)
        self.assertEqual(h.name, "House B")
        self.assertAlmostEqual(h.center_x, 3000.0)

    def test_get_nonexistent_house_returns_none(self):
        r = HouseRegistry(self.cfg)
        self.assertIsNone(r.get_house("Z"))


class TestSetTarget(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = _make_config(self.tmp, THREE_HOUSES)
        self.r = HouseRegistry(self.cfg)

    def test_set_valid_target(self):
        ok = self.r.set_target("A")
        self.assertTrue(ok)
        self.assertEqual(self.r.get_target_house().id, "A")

    def test_set_target_marks_in_progress(self):
        self.r.set_target("B")
        self.assertEqual(self.r.get_house("B").status, HouseStatus.IN_PROGRESS)

    def test_set_invalid_target(self):
        ok = self.r.set_target("NOTEXIST")
        self.assertFalse(ok)
        self.assertIsNone(self.r.get_target_house())

    def test_no_target_initially(self):
        self.assertIsNone(self.r.get_target_house())

    def test_switch_target(self):
        self.r.set_target("A")
        self.r.set_target("B")
        self.assertEqual(self.r.get_target_house().id, "B")
        # A should still be IN_PROGRESS (not reset)
        self.assertEqual(self.r.get_house("A").status, HouseStatus.IN_PROGRESS)


class TestMarkExplored(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = _make_config(self.tmp, THREE_HOUSES)
        self.r = HouseRegistry(self.cfg)

    def test_mark_explored_no_person(self):
        self.r.set_target("A")
        ok = self.r.mark_explored("A", person_found=False)
        self.assertTrue(ok)
        self.assertEqual(self.r.get_house("A").status, HouseStatus.EXPLORED)

    def test_mark_explored_with_person(self):
        self.r.set_target("B")
        loc = {"x": 3050.0, "y": 600.0, "z": 180.0}
        ok = self.r.mark_explored("B", person_found=True, person_location=loc)
        self.assertTrue(ok)
        self.assertEqual(self.r.get_house("B").status, HouseStatus.PERSON_FOUND)
        self.assertEqual(self.r.get_house("B").person_location, loc)

    def test_mark_explored_invalid_id(self):
        ok = self.r.mark_explored("NOTEXIST")
        self.assertFalse(ok)

    def test_mark_stores_notes(self):
        self.r.mark_explored("C", notes="Checked all rooms.")
        self.assertIn("Checked", self.r.get_house("C").notes)

    def test_mark_sets_end_time(self):
        self.r.set_target("A")
        self.r.mark_explored("A")
        self.assertIsNotNone(self.r.get_house("A").search_end_time)


class TestNearestUnsearched(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = _make_config(self.tmp, THREE_HOUSES)
        self.r = HouseRegistry(self.cfg)

    def test_nearest_to_A_is_A(self):
        # Standing right at A's center
        nearest = self.r.get_nearest_unsearched(1000, 500)
        self.assertEqual(nearest.id, "A")

    def test_nearest_to_B_is_B(self):
        nearest = self.r.get_nearest_unsearched(3000, 500)
        self.assertEqual(nearest.id, "B")

    def test_after_marking_A_explored_nearest_changes(self):
        self.r.mark_explored("A")
        nearest = self.r.get_nearest_unsearched(1000, 500)
        # B is at (3000,500), C at (2000,3000). From (1000,500): dist_B=2000, dist_C≈2693
        self.assertEqual(nearest.id, "B")

    def test_all_explored_returns_none(self):
        for h in self.r.get_all_houses():
            self.r.mark_explored(h.id)
        self.assertIsNone(self.r.get_nearest_unsearched(0, 0))

    def test_distance_calculation(self):
        # Manually verify distance from (0,0) to A(1000,500)
        expected = math.sqrt(1000**2 + 500**2)
        nearest = self.r.get_nearest_unsearched(0, 0)
        self.assertEqual(nearest.id, "A")
        self.assertAlmostEqual(nearest.distance_to(0, 0), expected, places=1)


class TestStatusSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = _make_config(self.tmp, THREE_HOUSES)
        self.r = HouseRegistry(self.cfg)

    def test_initial_summary(self):
        s = self.r.get_status_summary()
        self.assertEqual(s["UNSEARCHED"], 3)
        self.assertEqual(s.get("IN_PROGRESS", 0), 0)
        self.assertEqual(s.get("EXPLORED", 0), 0)

    def test_summary_after_operations(self):
        self.r.set_target("A")          # A → IN_PROGRESS
        self.r.mark_explored("A")       # A → EXPLORED
        self.r.set_target("B")          # B → IN_PROGRESS
        self.r.mark_explored("B", person_found=True)  # B → PERSON_FOUND
        s = self.r.get_status_summary()
        self.assertEqual(s["UNSEARCHED"], 1)
        self.assertEqual(s["EXPLORED"], 1)
        self.assertEqual(s["PERSON_FOUND"], 1)
        self.assertEqual(s.get("IN_PROGRESS", 0), 0)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = _make_config(self.tmp, THREE_HOUSES)

    def test_save_and_reload(self):
        r1 = HouseRegistry(self.cfg)
        r1.set_target("A")
        r1.mark_explored("A", person_found=False, notes="Saved test")
        r1.set_target("B")
        r1.save_to_file(self.cfg)

        r2 = HouseRegistry(self.cfg)
        self.assertEqual(r2.get_house("A").status, HouseStatus.EXPLORED)
        self.assertIn("Saved", r2.get_house("A").notes)
        self.assertEqual(r2.get_target_house().id, "B")
        self.assertEqual(r2.get_house("C").status, HouseStatus.UNSEARCHED)

    def test_reset_house(self):
        r = HouseRegistry(self.cfg)
        r.set_target("A")
        r.mark_explored("A")
        self.assertEqual(r.get_house("A").status, HouseStatus.EXPLORED)
        r.reset_house("A")
        self.assertEqual(r.get_house("A").status, HouseStatus.UNSEARCHED)

    def test_to_dict_is_serializable(self):
        r = HouseRegistry(self.cfg)
        r.set_target("A")
        d = r.to_dict()
        # Must be JSON-serializable
        serialized = json.dumps(d)
        self.assertIn("house_A" if "house_A" in serialized else "A", serialized)


# ──────────────────────────────────────────────
# Standalone runner (no pytest required)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
