from __future__ import annotations

import unittest
from types import SimpleNamespace

from showcase_api.control_plane import GuardState


def guard_config(**overrides):
    values = {
        "per_ip_window_seconds": 60,
        "per_ip_window_limit": 10,
        "per_ip_inflight_limit": 1,
        "global_inflight_limit": 1,
        "replay_nonce_ttl_seconds": 600,
        "replay_payload_ttl_seconds": 120,
        "replay_nonce_capacity": 2,
        "replay_payload_capacity": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class GuardStateTest(unittest.TestCase):
    def test_replay_cache_is_bounded(self):
        state = GuardState(guard_config())
        self.assertEqual(state.check_and_remember_replay("n1", "p1"), (True, None))
        self.assertEqual(state.check_and_remember_replay("n2", "p2"), (True, None))
        self.assertEqual(
            state.check_and_remember_replay("n3", "p3"),
            (False, "replay_cache_saturated"),
        )
        snapshot = state.snapshot()
        self.assertEqual(snapshot["replay_nonce_cache_size"], 2)
        self.assertEqual(snapshot["replay_payload_cache_size"], 2)

    def test_busy_request_can_roll_back_replay_reservation(self):
        state = GuardState(guard_config())
        self.assertEqual(state.check_and_remember_replay("n1", "p1"), (True, None))
        state.forget_replay("n1", "p1")
        self.assertEqual(state.check_and_remember_replay("n1", "p1"), (True, None))

    def test_duplicate_detection_remains_active(self):
        state = GuardState(guard_config())
        self.assertEqual(state.check_and_remember_replay("n1", "p1"), (True, None))
        self.assertEqual(state.check_and_remember_replay("n1", "different"), (False, "duplicate_nonce"))
        self.assertEqual(state.check_and_remember_replay("different", "p1"), (False, "duplicate_payload"))


if __name__ == "__main__":
    unittest.main()
