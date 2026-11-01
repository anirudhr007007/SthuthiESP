import unittest
from datetime import datetime

from greeting import GreetingController

FIXED_TIME = datetime(2026, 10, 25, 10, 30)


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def make_controller(**overrides):
    clock = overrides.pop("clock", FakeClock())
    return (
        GreetingController(
            "Malnad College of Engineering",
            presence_timeout=overrides.pop("presence_timeout", 8.0),
            visitor_welcome_cooldown=overrides.pop("visitor_welcome_cooldown", 20.0),
            min_greeting_gap=overrides.pop("min_greeting_gap", 2.0),
            clock=clock,
            wall_clock=lambda: FIXED_TIME,
        ),
        clock,
    )


class KnownPersonTests(unittest.TestCase):
    def test_greets_a_confirmed_person_once_per_visit(self):
        controller, clock = make_controller()
        actions = controller.update({"Alice"}, False, now=clock.value)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "known")
        self.assertIn("Alice", actions[0].message)
        self.assertIn("10:30 AM", actions[0].message)

        # A second batch during the same visit must not repeat the greeting.
        self.assertEqual(controller.update({"Alice"}, False, now=1.0), [])

    def test_absent_person_is_greeted_again_after_timeout(self):
        controller, _ = make_controller(presence_timeout=5.0, min_greeting_gap=1.0)
        self.assertEqual(len(controller.update({"Alice"}, False, now=0.0)), 1)
        self.assertEqual(controller.update({"Alice"}, False, now=1.0), [])
        # No face for longer than the presence timeout ends the visit.
        self.assertEqual(controller.update(set(), False, now=7.0), [])
        self.assertEqual(len(controller.update({"Alice"}, False, now=8.0)), 1)

    def test_minimum_greeting_gap_applies_between_different_people(self):
        controller, _ = make_controller(min_greeting_gap=5.0)
        self.assertEqual(len(controller.update({"Alice"}, False, now=0.0)), 1)
        self.assertEqual(controller.update({"Bob"}, False, now=2.0), [])
        actions = controller.update({"Bob"}, False, now=5.0)
        self.assertEqual(len(actions), 1)
        self.assertIn("Bob", actions[0].message)

    def test_names_are_deduplicated_and_unknown_is_ignored(self):
        controller, _ = make_controller()
        actions = controller.update(
            ["Alice", "Alice", "Unknown", "  Alice  ", "", "   "], False, now=0.0
        )
        self.assertEqual(len(actions), 1)
        self.assertIn("Alice", actions[0].message)


class VisitorTests(unittest.TestCase):
    def test_unknown_visitor_is_welcomed_once(self):
        controller, _ = make_controller()
        actions = controller.update(set(), True, now=0.0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "visitor")
        self.assertIn("Welcome to Malnad College of Engineering", actions[0].message)
        self.assertEqual(controller.update(set(), True, now=1.0), [])

    def test_visitor_welcome_obeys_cooldown_after_a_new_visit(self):
        controller, _ = make_controller(
            presence_timeout=5.0, visitor_welcome_cooldown=20.0
        )
        self.assertEqual(len(controller.update(set(), True, now=0.0)), 1)
        # The unknown face leaves, so the welcome flag resets.
        self.assertEqual(controller.update(set(), False, now=7.0), [])
        # It returns before the cooldown has elapsed.
        self.assertEqual(controller.update(set(), True, now=10.0), [])
        # After the cooldown, a new visitor can be welcomed.
        self.assertEqual(len(controller.update(set(), True, now=21.0)), 1)

    def test_known_greeting_wins_when_known_and_unknown_are_present(self):
        controller, _ = make_controller(min_greeting_gap=0.0)
        actions = controller.update({"Alice"}, True, now=0.0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "known")
        # The visitor welcome follows on the next update.
        actions = controller.update({"Alice"}, True, now=1.0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "visitor")

    def test_reset_clears_visit_state(self):
        controller, clock = make_controller()
        controller.update({"Alice"}, True, now=0.0)
        controller.reset()
        actions = controller.update({"Alice"}, True, now=1.0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].kind, "known")


if __name__ == "__main__":
    unittest.main()
