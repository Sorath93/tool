# unit testing

import unittest
from analyzer import SentimentAnalyzer, ModelLoadError


class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SentimentAnalyzer()

    def test_positive_sentiment(self):
        text = "I love this product! It's amazing."
        score = self.analyzer.analyze(text)[0]
        label = self.analyzer.analyze(text)[1]
        self.assertGreater(score, 0)
        self.assertEqual(label, "POSITIVE")

    def test_negative_sentiment(self):
        text = "This is the worst experience I've ever had."
        score = self.analyzer.analyze(text)[0]
        label = self.analyzer.analyze(text)[1]
        self.assertLess(score, 0)
        self.assertEqual(label, "NEGATIVE")

    def test_neutral_sentiment(self):
        text = "The product is okay nothing special."
        score = self.analyzer.analyze(text)[0]
        label = self.analyzer.analyze(text)[1]
        print(score, label)
        self.assertAlmostEqual(score, 0, delta=0.5)
        self.assertEqual(label, "NEUTRAL")

    def test_invalid_inputs(self):
        invalid_inputs = ["", "   ", None, 123, [], {}]
        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                with self.assertRaises(ValueError):
                    self.analyzer.analyze(invalid_input)

    def test_very_long_text(self):
        text = "good " * 1000
        score = self.analyzer.analyze(text)[0]
        label = self.analyzer.analyze(text)[1]
        self.assertGreater(score, 0)
        self.assertEqual(label, "POSITIVE")

    def test_punctuation_only(self):
        text = "!!!"
        score = self.analyzer.analyze(text)[0]
        label = self.analyzer.analyze(text)[1]
        self.assertGreater(score, 0)
        self.assertEqual(label, "POSITIVE")

    def test_emojis_only(self):
        text = "😀😞👍👎"
        score = self.analyzer.analyze(text)[0]
        label = self.analyzer.analyze(text)[1]
        self.assertAlmostEqual(score, 0, delta=0.5)
        self.assertEqual(label, "NEUTRAL")

    def test_preprocess_text(self):
        text = "Check this out!!! https://example.com @user"
        processed = self.analyzer._preprocess_text(text)
        expected = "Check this out!!! http @user"
        self.assertEqual(processed, expected)

    def test_preprocess_text_case_sensitivity(self):
        text = "HTTP://Example.COM"
        processed = self.analyzer._preprocess_text(text)
        expected = text
        self.assertEqual(processed, expected)

    def test_model_load_failure(self):
        with self.assertRaises(ModelLoadError):
            SentimentAnalyzer(model_name="non_existent_model")
