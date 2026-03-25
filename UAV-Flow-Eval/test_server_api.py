"""
Layer 2 — Server API tests with a lightweight stub server.
No Unreal Engine / gym_unrealcv needed.

The stub replaces BasicUAVControlBackend with a fake that returns
predictable data, so we can test all HTTP routes and house-registry
integration without touching the simulator.

Run:
    cd UAV-Flow-Eval
    python test_server_api.py
  or:
    python -m pytest test_server_api.py -v
"""
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib import request as urlrequest

from house_registry import HouseRegistry, HouseStatus


# ──────────────────────────────────────────────
# Minimal stub server  (mirrors the real server's routes)
# ──────────────────────────────────────────────

_FAKE_POSE = {"x": 2400.0, "y": 100.0, "z": 200.0, "task_yaw": -90.0, "uav_yaw": -90.0, "command_yaw": -90.0}
_FAKE_DEPTH = {"available": True, "front_min_depth": 180.0, "front_mean_depth": 240.0,
               "min_depth": 50.0, "max_depth": 900.0, "image_width": 640, "image_height": 480}

THREE_HOUSES = [
    {"id": "house_A", "name": "House A", "center_x": 2400.0, "center_y": 100.0,
     "center_z": 200.0, "approach_z": 600.0, "radius_cm": 700.0, "entry_yaw_hint": -90.0},
    {"id": "house_B", "name": "House B", "center_x": 3800.0, "center_y": 800.0,
     "center_z": 200.0, "approach_z": 600.0, "radius_cm": 750.0, "entry_yaw_hint": 180.0},
    {"id": "house_C", "name": "House C", "center_x": 2100.0, "center_y": 2200.0,
     "center_z": 200.0, "approach_z": 600.0, "radius_cm": 680.0, "entry_yaw_hint": 0.0},
]


def _make_stub_config(tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "stub_houses.json")
    with open(path, "w") as f:
        json.dump({"world_bounds": {"min_x": 1000, "min_y": -500,
                                    "max_x": 5000, "max_y": 3000},
                   "houses": THREE_HOUSES}, f)
    return path


class StubBackend:
    """Minimal replacement for BasicUAVControlBackend."""

    def __init__(self, cfg_path: str):
        self.house_registry = HouseRegistry(cfg_path)
        self._cfg_path = cfg_path
        self.last_action = "idle"
        self.movement_enabled = True
        self._move_log: list = []

    def get_state(self, *, status="ok", message="") -> Dict[str, Any]:
        return {
            "status": status, "message": message,
            "task_label": "test task",
            "movement_enabled": self.movement_enabled,
            "last_action": self.last_action,
            "pose": dict(_FAKE_POSE),
            "depth": dict(_FAKE_DEPTH),
            "camera_info": {},
            "last_capture": None,
            "house_registry": self.house_registry.get_status_summary(),
        }

    def get_house_registry(self) -> Dict[str, Any]:
        return {"status": "ok", "registry": self.house_registry.to_dict()}

    def select_target_house(self, house_id: str) -> Dict[str, Any]:
        ok = self.house_registry.set_target(house_id)
        if not ok:
            return {"status": "error", "message": f"House '{house_id}' not found."}
        self.house_registry.save_to_file(self._cfg_path)
        house = self.house_registry.get_house(house_id)
        return {"status": "ok", "message": f"Target set to '{house_id}'.",
                "target_house": house.to_dict() if house else None,
                "registry_summary": self.house_registry.get_status_summary()}

    def mark_house_explored(self, house_id: str, *, person_found=False,
                             person_location=None, notes="") -> Dict[str, Any]:
        ok = self.house_registry.mark_explored(house_id, person_found=person_found,
                                                person_location=person_location, notes=notes)
        if not ok:
            return {"status": "error", "message": f"House '{house_id}' not found."}
        self.house_registry.save_to_file(self._cfg_path)
        return {"status": "ok",
                "message": f"House '{house_id}' marked {'PERSON_FOUND' if person_found else 'EXPLORED'}.",
                "registry_summary": self.house_registry.get_status_summary()}

    def navigate_step_to_house(self, house_id: str) -> Dict[str, Any]:
        house = self.house_registry.get_house(house_id)
        if house is None:
            return {"status": "error", "message": f"House '{house_id}' not found."}
        self._move_log.append({"action": "nav_step", "target": house_id})
        self.last_action = "nav_forward"
        return self.get_state(message=f"Nav step toward {house_id}.")

    def move_relative(self, *, forward_cm=0.0, right_cm=0.0, up_cm=0.0,
                      yaw_delta_deg=0.0, action_name="custom") -> Dict[str, Any]:
        self._move_log.append({"action": action_name,
                                "forward_cm": forward_cm, "right_cm": right_cm,
                                "up_cm": up_cm, "yaw_delta_deg": yaw_delta_deg})
        self.last_action = action_name
        return self.get_state(message=f"Executed {action_name}.")

    def set_movement_enabled(self, enabled: bool) -> Dict[str, Any]:
        self.movement_enabled = enabled
        return self.get_state(message="Movement toggled.")

    def capture_frame(self, label=None) -> Dict[str, Any]:
        return {"status": "ok", "capture": {"capture_id": "test_capture"},
                "state": self.get_state()}


def _make_stub_handler(backend: StubBackend):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload, code=200):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self):
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0:
                return {}
            raw = self.rfile.read(n)
            try:
                d = json.loads(raw)
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}

        def log_message(self, *_):
            pass  # suppress output during tests

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/health"):
                self._send_json({"status": "ok"})
            elif path == "/state":
                self._send_json(backend.get_state())
            elif path == "/house_registry":
                self._send_json(backend.get_house_registry())
            elif path == "/frame":
                # Return a 1x1 white JPEG
                import struct, zlib
                body = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
                body += b"\xff\xdb\x00C\x00" + bytes([8]*64)
                body += b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xda\x00\x08"
                body += b"\x01\x01\x00\x00?\x00\xf8\xff\xd9"
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_json({"status": "error", "message": "not found"}, 404)

        def do_POST(self):
            path = self.path.split("?")[0]
            data = self._read_body()
            if path == "/move_relative":
                self._send_json(backend.move_relative(
                    forward_cm=float(data.get("forward_cm", 0)),
                    right_cm=float(data.get("right_cm", 0)),
                    up_cm=float(data.get("up_cm", 0)),
                    yaw_delta_deg=float(data.get("yaw_delta_deg", 0)),
                    action_name=str(data.get("action_name", "custom")),
                ))
            elif path == "/basic_movement_enable":
                self._send_json(backend.set_movement_enabled(bool(data.get("enabled", False))))
            elif path == "/capture":
                self._send_json(backend.capture_frame(label=data.get("label")))
            elif path == "/select_target_house":
                hid = str(data.get("house_id", ""))
                if not hid:
                    self._send_json({"status": "error", "message": "house_id required."}, 400)
                else:
                    self._send_json(backend.select_target_house(hid))
            elif path == "/mark_house_explored":
                hid = str(data.get("house_id", ""))
                if not hid:
                    self._send_json({"status": "error", "message": "house_id required."}, 400)
                else:
                    self._send_json(backend.mark_house_explored(
                        hid,
                        person_found=bool(data.get("person_found", False)),
                        person_location=data.get("person_location"),
                        notes=str(data.get("notes", "")),
                    ))
            elif path == "/navigate_step_to_house":
                hid = str(data.get("house_id", ""))
                target = backend.house_registry.get_target_house()
                if not hid and target:
                    hid = target.id
                if not hid:
                    self._send_json({"status": "error", "message": "no house_id"}, 400)
                else:
                    self._send_json(backend.navigate_step_to_house(hid))
            elif path == "/refresh":
                self._send_json(backend.get_state(message="refreshed"))
            elif path == "/task":
                self._send_json({"status": "ok", "task_label": data.get("task_label", "")})
            elif path == "/shutdown":
                self._send_json({"status": "ok"})
            else:
                self._send_json({"status": "error", "message": "not found"}, 404)

    return Handler


# ──────────────────────────────────────────────
# Test base: starts stub server, stops after each test class
# ──────────────────────────────────────────────

class StubServerTestCase(unittest.TestCase):
    """Base class: spins up a stub HTTP server on a free port."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.cfg_path = _make_stub_config(cls.tmp)
        cls.backend = StubBackend(cls.cfg_path)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_stub_handler(cls.backend))
        cls.port = cls.httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls._thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def _get(self, path) -> Dict[str, Any]:
        with urlrequest.urlopen(f"{self.base}{path}", timeout=5) as r:
            return json.loads(r.read())

    def _post(self, path, payload=None) -> Dict[str, Any]:
        body = json.dumps(payload or {}).encode()
        req = urlrequest.Request(
            f"{self.base}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urlrequest.urlopen(req, timeout=5) as r:
            return json.loads(r.read())

    def _reset_registry(self):
        """Reset all houses to UNSEARCHED between tests."""
        for h in self.backend.house_registry.get_all_houses():
            self.backend.house_registry.reset_house(h.id)
        self.backend.house_registry._current_target_id = ""
        self.backend._move_log.clear()


# ──────────────────────────────────────────────
# Test: basic endpoints
# ──────────────────────────────────────────────

class TestBasicEndpoints(StubServerTestCase):
    def test_health(self):
        r = self._get("/health")
        self.assertEqual(r["status"], "ok")

    def test_state_has_pose(self):
        r = self._get("/state")
        self.assertIn("pose", r)
        self.assertAlmostEqual(r["pose"]["x"], 2400.0)

    def test_state_has_house_registry(self):
        r = self._get("/state")
        self.assertIn("house_registry", r)
        self.assertIn("UNSEARCHED", r["house_registry"])

    def test_move_relative(self):
        r = self._post("/move_relative", {"forward_cm": 25.0, "action_name": "forward"})
        self.assertEqual(r["status"], "ok")
        self.assertEqual(self.backend.last_action, "forward")

    def test_capture(self):
        r = self._post("/capture", {"label": "test"})
        self.assertEqual(r["status"], "ok")
        self.assertIn("capture", r)

    def test_frame_returns_jpeg(self):
        req = urlrequest.Request(f"{self.base}/frame")
        with urlrequest.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.headers["Content-Type"], "image/jpeg")

    def test_404_unknown_route(self):
        from urllib.error import HTTPError
        with self.assertRaises(HTTPError) as ctx:
            self._get("/nonexistent_route")
        self.assertEqual(ctx.exception.code, 404)


# ──────────────────────────────────────────────
# Test: house registry endpoints
# ──────────────────────────────────────────────

class TestHouseRegistryEndpoints(StubServerTestCase):
    def setUp(self):
        self._reset_registry()

    def test_get_house_registry(self):
        r = self._get("/house_registry")
        self.assertEqual(r["status"], "ok")
        houses = r["registry"]["houses"]
        self.assertEqual(len(houses), 3)
        ids = {h["id"] for h in houses}
        self.assertEqual(ids, {"house_A", "house_B", "house_C"})

    def test_select_target_house_valid(self):
        r = self._post("/select_target_house", {"house_id": "house_A"})
        self.assertEqual(r["status"], "ok")
        self.assertIn("house_A", r["message"])
        # Verify state reflects the selection
        state = self._get("/state")
        self.assertEqual(state["house_registry"]["target_house_id"], "house_A")

    def test_select_target_house_invalid(self):
        r = self._post("/select_target_house", {"house_id": "NOTEXIST"})
        self.assertEqual(r["status"], "error")

    def test_select_target_house_missing_id(self):
        from urllib.error import HTTPError
        with self.assertRaises(HTTPError) as ctx:
            self._post("/select_target_house", {})
        self.assertEqual(ctx.exception.code, 400)

    def test_mark_house_explored(self):
        self._post("/select_target_house", {"house_id": "house_B"})
        r = self._post("/mark_house_explored", {"house_id": "house_B", "person_found": False})
        self.assertEqual(r["status"], "ok")
        self.assertEqual(
            self.backend.house_registry.get_house("house_B").status,
            HouseStatus.EXPLORED,
        )

    def test_mark_house_person_found(self):
        self._post("/select_target_house", {"house_id": "house_C"})
        r = self._post("/mark_house_explored", {
            "house_id": "house_C",
            "person_found": True,
            "person_location": {"x": 2100.0, "y": 2200.0, "z": 150.0},
            "notes": "Found near kitchen",
        })
        self.assertEqual(r["status"], "ok")
        h = self.backend.house_registry.get_house("house_C")
        self.assertEqual(h.status, HouseStatus.PERSON_FOUND)
        self.assertIsNotNone(h.person_location)

    def test_mark_house_invalid_id(self):
        r = self._post("/mark_house_explored", {"house_id": "NOTEXIST"})
        self.assertEqual(r["status"], "error")

    def test_registry_summary_updates_after_operations(self):
        self._post("/select_target_house", {"house_id": "house_A"})
        self._post("/mark_house_explored", {"house_id": "house_A"})
        state = self._get("/state")
        s = state["house_registry"]
        self.assertEqual(s.get("EXPLORED", 0), 1)
        self.assertEqual(s.get("UNSEARCHED", 0), 2)


# ──────────────────────────────────────────────
# Test: navigation step endpoint
# ──────────────────────────────────────────────

class TestNavigateToHouse(StubServerTestCase):
    def setUp(self):
        self._reset_registry()

    def test_navigate_step_with_house_id(self):
        self._post("/select_target_house", {"house_id": "house_B"})
        r = self._post("/navigate_step_to_house", {"house_id": "house_B"})
        self.assertEqual(r["status"], "ok")
        self.assertGreater(len(self.backend._move_log), 0)

    def test_navigate_step_uses_target_when_no_id(self):
        self._post("/select_target_house", {"house_id": "house_A"})
        self.backend._move_log.clear()
        r = self._post("/navigate_step_to_house", {})   # no house_id → use target
        self.assertEqual(r["status"], "ok")

    def test_navigate_step_no_target_no_id_returns_400(self):
        from urllib.error import HTTPError
        with self.assertRaises(HTTPError) as ctx:
            self._post("/navigate_step_to_house", {})
        self.assertEqual(ctx.exception.code, 400)

    def test_navigate_step_invalid_house(self):
        r = self._post("/navigate_step_to_house", {"house_id": "INVALID"})
        self.assertEqual(r["status"], "error")


# ──────────────────────────────────────────────
# Test: full workflow simulation
# ──────────────────────────────────────────────

class TestFullMissionWorkflow(StubServerTestCase):
    """
    Simulates a complete multi-house search mission through the API.
    """

    def setUp(self):
        self._reset_registry()

    def test_full_3_house_mission(self):
        houses = ["house_A", "house_B", "house_C"]

        for i, hid in enumerate(houses):
            # 1. Select target
            r = self._post("/select_target_house", {"house_id": hid})
            self.assertEqual(r["status"], "ok", f"Failed to select {hid}")

            # 2. Simulate a few nav steps
            for _ in range(3):
                r = self._post("/navigate_step_to_house", {"house_id": hid})
                self.assertEqual(r["status"], "ok")

            # 3. Mark result: person found in last house, not found in others
            person_found = (i == len(houses) - 1)
            r = self._post("/mark_house_explored", {
                "house_id": hid,
                "person_found": person_found,
                "person_location": {"x": 2100.0, "y": 2200.0, "z": 150.0} if person_found else None,
                "notes": f"Searched house {i+1}",
            })
            self.assertEqual(r["status"], "ok")

        # Final state check
        reg = self.backend.house_registry
        explored = [h for h in reg.get_all_houses()
                    if h.status == HouseStatus.EXPLORED]
        found = [h for h in reg.get_all_houses()
                 if h.status == HouseStatus.PERSON_FOUND]
        self.assertEqual(len(explored), 2)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].id, "house_C")

        # No unsearched houses should remain
        s = reg.get_status_summary()
        self.assertEqual(s.get("UNSEARCHED", 0), 0)

    def test_auto_select_nearest_via_registry_logic(self):
        """
        Verify that after marking house_A explored, the nearest unsearched
        from house_A's position is house_B (not house_C which is farther).
        """
        self._post("/mark_house_explored", {"house_id": "house_A"})
        nearest = self.backend.house_registry.get_nearest_unsearched(2400.0, 100.0)
        self.assertIsNotNone(nearest)
        # house_B(3800,800): dist=sqrt(1400^2+700^2)≈1565  house_C(2100,2200): dist≈2110
        self.assertEqual(nearest.id, "house_B")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
