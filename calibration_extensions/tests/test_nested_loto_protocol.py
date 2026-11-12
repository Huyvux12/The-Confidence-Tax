"""Fast regression checks for the optional nested LOTO protocol.

No inference, SLSQP fitting, or CMA-ES optimisation is executed by these tests.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import v4_study
import evo_study


class FakeSources:
    def __init__(self, labels):
        self.source_id = labels

    def sources(self):
        return list(dict.fromkeys(self.source_id))


class NestedLotoTests(unittest.TestCase):
    def test_heldout_source_never_enters_spec_selection(self):
        tr = FakeSources([f"source_{i}" for i in range(24)])
        selected = SimpleNamespace(kind="rbf_oct", J=4, sigma=0.5, lam=0.01)
        with patch.object(v4_study, "menu", return_value=[
            ("constant", 0, None, 0.0),
            ("rbf_oct", 4, 0.5, 0.01),
        ]):
            with patch("calibration_extensions.rb_calibration.select_by_loso",
                       return_value=(selected, [])) as choose:
                spec = v4_study.nested_loto_spec(tr, "v4b", "heldout")
        self.assertEqual(spec, ["rbf_oct", 4, 0.5, 0.01])
        self.assertIs(choose.call_args.args[0], tr)
        self.assertEqual(len(choose.call_args.args[1]), 2)
        self.assertEqual(choose.call_args.args[2:], (0.01, 0.01, 1.645))

    def test_selector_rejects_leaked_source(self):
        tr = FakeSources(["heldout"] + [f"source_{i}" for i in range(23)])
        with self.assertRaisesRegex(ValueError, "leaked"):
            v4_study.nested_loto_spec(tr, "v4b", "heldout")

    def test_selector_rejects_wrong_number_of_sources(self):
        tr = FakeSources([f"source_{i}" for i in range(23)])
        with self.assertRaisesRegex(ValueError, "24"):
            v4_study.nested_loto_spec(tr, "v4b", "heldout")

    def test_nested_job_generation_does_not_read_global_chosen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs/gate_study").mkdir(parents=True)
            (root / "runs/gate_study/hypers.json").write_text(
                json.dumps({"units": {"model|prompt": {}}, "tasks": ["a", "b"]}),
                encoding="utf-8",
            )
            # No runs/v4_study/chosen.json exists, intentionally.
            with patch.object(v4_study, "ROOT", root), patch.object(evo_study, "ROOT", root):
                v4 = v4_study.jobs_for("loto_nested", variants=["v4b"])
                evo = evo_study.jobs_for("evo_loto_nested")
        self.assertEqual(len(v4), 2)
        self.assertEqual(len(evo), 2)
        self.assertTrue(all("spec" not in x for x in v4 + evo))
        self.assertTrue(all(x["stage"].endswith("nested") for x in v4 + evo))

    def test_nested_evo_dispatch_never_accepts_global_spec(self):
        job = {"stage": "evo_loto_nested", "unit": ["model", "prompt"],
               "task": "heldout", "variant": "evo_rb"}
        with patch.object(evo_study, "evo_loto_job", return_value={"ok": True}) as run:
            self.assertEqual(evo_study.run_job(job), {"ok": True})
        self.assertEqual(run.call_args.kwargs, {"nested": True})
        self.assertEqual(run.call_args.args, (("model", "prompt"), "heldout"))


if __name__ == "__main__":
    unittest.main()
