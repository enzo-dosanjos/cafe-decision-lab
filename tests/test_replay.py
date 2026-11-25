"""Validate replay integrity, fixture transitions and overflow labels."""

from importlib.util import find_spec
import unittest

from cafe.replay import demo, frame, validate


class ReplayTests(unittest.TestCase):
    """Exercise the offline replay contract."""

    def test_invalid_bundle(self):
        """Reject wrong lineage, coverage, counts and product order."""
        for field, value in [("minute", 0), ("queue", -1), ("staff", 1.5), ("stock", [17]*3)]:
            bundle = demo()
            bundle["runs"][2]["minutes"][1][field] = value

            with self.assertRaisesRegex(ValueError, "mlp.*minute 1"):
                validate(bundle)

        bundle = demo()
        bundle["runs"][0]["scenario"] = "other"

        with self.assertRaisesRegex(ValueError, "fixed: mismatched"):
            validate(bundle)

        bundle = demo()
        bundle["runs"][1]["minutes"].pop()

        with self.assertRaisesRegex(ValueError, "adaptive: incomplete"):
            validate(bundle)

    @unittest.skipUnless(find_spec("matplotlib"), "requires report extra")
    def test_transition_and_overflow(self):
        """Show exact queues, closing stockouts and synchronized new-period counts."""
        bundle = demo()
        validate(bundle)
        bundle["runs"][0]["minutes"][10]["queue"] = 12
        before = frame(bundle, 10).figure.axes[0]
        labels = {t.get_text() for t in before.texts}
        self.assertTrue({"12:10", "+4", "12 waiting", "0 left this period"} <= labels)
        after = frame(bundle, 11).figure.axes[0]
        labels = [t.get_text() for t in after.texts]
        self.assertEqual(labels.count("1 on duty"), 4)
        self.assertEqual(labels.count("0 left this period"), 4)
        self.assertEqual(labels.count("10"), 12)
        self.assertIn("12:11", labels)
