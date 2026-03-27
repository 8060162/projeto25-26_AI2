"""Regression tests for shared text utility folding helpers."""

from __future__ import annotations

import unittest

from Chunking.utils.text import fold_editorial_text


class TextUtilityRegressionTests(unittest.TestCase):
    """Protect reusable editorial folding behavior across layers."""

    def test_fold_editorial_text_preserves_word_boundaries_when_requested(self) -> None:
        """Ensure normalizer-style folding keeps token boundaries stable."""
        folded = fold_editorial_text(
            "Diário da República | N.º 120",
            preserve_word_boundaries=True,
        )

        self.assertEqual(folded, "diario da republica n o 120")

    def test_fold_editorial_text_can_return_compact_signature(self) -> None:
        """Ensure validator-style folding still supports compact matching."""
        folded = fold_editorial_text(
            "Diário da República | N.º 120",
            preserve_word_boundaries=False,
        )

        self.assertEqual(folded, "diariodarepublicano120")


if __name__ == "__main__":
    unittest.main()
