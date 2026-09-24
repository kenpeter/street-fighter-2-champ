import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import loop_sf2
import ryu_inferred_replay as replay
import sched_chain_sf2 as scheduler


class SchedulerResumeTests(unittest.TestCase):
    def test_resumes_active_attempt_before_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "chain_status.json"
            status = {
                "done": [],
                "failed": [],
                "reflections": {
                    "ken_ryu_4": {
                        "outcome": "LOSS",
                        "attempt": 4,
                        "suggestion": "more-damage-zone",
                    }
                },
                "active": {
                    "stage": "ken_ryu_4",
                    "attempt": 5,
                    "strategy": "base",
                    "delay": 30,
                    "seed": 1,
                    "max_frames": 9000,
                    "tag": "ken_ryu_4_a5",
                },
            }
            status_path.write_text(json.dumps(status))
            observed = []
            calls = []

            def fake_run_play(stage, **kwargs):
                calls.append((stage, kwargs))
                observed.append(json.loads(status_path.read_text())["active"])
                if len(calls) == 1:
                    return "LOSS", 120, {
                        "outcome": "LOSS",
                        "hp": 0,
                        "ehp": 100,
                        "mw": 0,
                        "emw": 1,
                        "ticks": 120,
                        "history_tail": [],
                    }
                return "WIN", 130, {
                    "outcome": "WIN",
                    "hp": 100,
                    "ehp": 0,
                    "mw": 1,
                    "emw": 0,
                    "ticks": 130,
                    "history_tail": [],
                }

            with patch.object(scheduler, "STATUS", str(status_path)), patch.object(
                scheduler, "log"
            ), patch.object(scheduler, "run_play", side_effect=fake_run_play):
                state = scheduler.load_status_resume()
                result = scheduler.do_stage("ken_ryu_4", object(), state)

            self.assertTrue(result)
            self.assertEqual(calls[0][0], "ken_ryu_4")
            self.assertEqual(calls[0][1]["tag"], "ken_ryu_4_a5")
            self.assertEqual(calls[0][1]["strategy"], "base")
            self.assertEqual(calls[0][1]["delay"], 30)
            self.assertEqual(calls[0][1]["seed"], 1)
            self.assertEqual(observed[0]["attempt"], 5)
            saved = json.loads(status_path.read_text())
            self.assertIsNone(saved["active"])
            self.assertIn("ken_ryu_4", saved["done"])

    def test_uses_next_attempt_after_saved_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "chain_status.json"
            status = {
                "done": [],
                "failed": [],
                "reflections": {
                    "ken_ryu_4": {
                        "outcome": "LOSS",
                        "attempt": 4,
                        "suggestion": "more-damage-zone",
                    }
                },
            }
            status_path.write_text(json.dumps(status))
            calls = []

            def fake_run_play(stage, **kwargs):
                calls.append((stage, kwargs))
                return "WIN", 130, {
                    "outcome": "WIN",
                    "hp": 100,
                    "ehp": 0,
                    "mw": 1,
                    "emw": 0,
                    "ticks": 130,
                    "history_tail": [],
                }

            with patch.object(scheduler, "STATUS", str(status_path)), patch.object(
                scheduler, "log"
            ), patch.object(scheduler, "run_play", side_effect=fake_run_play):
                state = scheduler.load_status_resume()
                result = scheduler.do_stage("ken_ryu_4", object(), state)

            self.assertTrue(result)
            self.assertEqual(calls[0][1]["tag"], "ken_ryu_4_a5")
            self.assertEqual(calls[0][1]["strategy"], "base")
            self.assertEqual(calls[0][1]["delay"], 30)
            self.assertEqual(calls[0][1]["seed"], 1)

    def test_inferred_replay_maps_tatsu_motion(self):
        state = {
            "px": 100,
            "ex": 180,
            "pstatus": 512,
            "estatus": 512,
            "ram_e_proj": 0,
        }
        segments = replay.move_segments("tatsu", state, 0, "none")
        self.assertEqual(
            segments,
            [("tatsu_down", 3), ("tatsu_diag", 4), ("tatsu_punch", 3), ("tatsu_finish", 1)],
        )
        action, why = replay.action_for("tatsu_punch", state, 0)
        self.assertEqual(action, replay.A_C_LEFT)
        self.assertEqual(why, "tatsu_punch")

    def test_loop_prepare_preserves_loser_and_active_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status_path = root / "chain_status.json"
            status = {
                "done": ["ken_chunli_1"],
                "failed": ["ken_ryu_4"],
                "reflections": {
                    "ken_ryu_4": {"outcome": "LOSS", "attempt": 5}
                },
                "active": {"stage": "ken_ryu_4", "attempt": 6},
            }
            status_path.write_text(json.dumps(status))
            with patch.object(loop_sf2, "ROOT", root), patch.object(loop_sf2, "log"):
                loop_sf2.prepare_chain_resume()
            saved = json.loads(status_path.read_text())
            self.assertEqual(saved["failed"], [])
            self.assertEqual(saved["reflections"]["ken_ryu_4"]["attempt"], 5)
            self.assertEqual(saved["active"]["attempt"], 6)


if __name__ == "__main__":
    unittest.main()
