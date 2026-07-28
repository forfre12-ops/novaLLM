"""Unit checks for resumable training checkpoint helpers.

These tests avoid GPU/model loading; the actual 4bit resume path is verified
manually with a tiny training config when CUDA is available.
"""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_sft", ROOT / "scripts" / "02_train_sft.py")
assert SPEC and SPEC.loader
train_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_sft)


def test_resolve_latest_checkpoint_uses_numeric_order():
    with tempfile.TemporaryDirectory(prefix="nova-resume-test-") as tmp:
        root = Path(tmp)
        (root / "checkpoint-2").mkdir()
        (root / "checkpoint-10").mkdir()
        (root / "checkpoint-bad").mkdir()

        latest = train_sft.resolve_resume_path("latest", root)
        assert latest.name == "checkpoint-10"


def test_resume_latest_missing_raises_system_exit():
    with tempfile.TemporaryDirectory(prefix="nova-resume-test-") as tmp:
        try:
            train_sft.resolve_resume_path("latest", Path(tmp))
        except SystemExit as exc:
            assert "no checkpoints found" in str(exc)
        else:
            raise AssertionError("expected SystemExit for missing checkpoints")


def test_move_optimizer_state_to_device_replaces_tensor_like_values():
    class FakeTensor:
        def __init__(self):
            self.device = None

        def to(self, device):
            moved = FakeTensor()
            moved.device = device
            return moved

    class FakeOptimizer:
        def __init__(self):
            self.state = {0: {"step": 1, "exp_avg": FakeTensor()}}

    opt = FakeOptimizer()
    train_sft.move_optimizer_state_to_device(opt, "cuda")

    assert opt.state[0]["step"] == 1
    assert opt.state[0]["exp_avg"].device == "cuda"


def main() -> int:
    tests = [
        test_resolve_latest_checkpoint_uses_numeric_order,
        test_resume_latest_missing_raises_system_exit,
        test_move_optimizer_state_to_device_replaces_tensor_like_values,
    ]
    failed = []
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
        except Exception as exc:
            failed.append((test.__name__, exc))
            print(f"  [FAIL] {test.__name__}: {exc}")
    print(f"\ntest_train_resume: {len(tests) - len(failed)}/{len(tests)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
