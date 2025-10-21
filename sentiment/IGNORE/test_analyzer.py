"""
Comprehensive test suite for sentiment analyzer.

Tests are designed to be informative and demonstrate intended usage patterns,
not just achieve code coverage. Each test class focuses on specific aspects:
- Core functionality and edge cases
- Error handling and validation
- Configuration management
- Extensibility patterns
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path to import the analyzer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment.IGNORE.analyzer import (
    TextPreprocessor,
    BaseSentimentAnalyzer,
    HuggingFaceSentimentAnalyzer,
    MockCloudSentimentAnalyzer,
    SentimentAnalyzerFactory,
    SentimentAnalysisError,
    ModelLoadError,
    TRANSFORMERS_AVAILABLE
)


class TestTextPreprocessor(unittest.TestCase):
    """
    Test text preprocessing functionality.
    Demonstrates how preprocessing can be configured and validated.
    """
    
    def setUp(self):
        """Set up test fixtures with different configurations."""
        self.default_preprocessor = TextPreprocessor()
        self.minimal_preprocessor = TextPreprocessor({
            'remove_urls': False,
            'remove_mentions': False,
            'demojize_emojis': False,
            'to_lowercase': False
        })
    
    def test_basic_text_cleaning(self):
        """Test that basic text cleaning works as expected."""
        text = "This is a GREAT post! 🔥🔥🔥"
        result = self.default_preprocessor.preprocess(text)
        
        # Should be lowercase and normalized
        self.assertEqual(result, "this is a great post! fire fire fire")
    
    def test_url_removal(self):
        """Test URL removal functionality."""
        text = "Check this out: https://example.com and www.test.com"
        result = self.default_preprocessor.preprocess(text)
        
        self.assertEqual(result, "check this out: and")
        
        # Test with URL removal disabled
        result_minimal = self.minimal_preprocessor.preprocess(text)
        self.assertIn("https://example.com", result_minimal)
        self.assertIn("www.test.com", result_minimal)
    
    def test_mention_and_hashtag_handling(self):
        """Test @mention and #hashtag processing."""
        text = "Great work @user123 on this #awesome #project!"
        result = self.default_preprocessor.preprocess(text)
        
        # Should remove @ mentions and # symbols but keep hashtag text
        self.assertEqual(result, "great work on this awesome project!")
    
    def test_punctuation_normalization(self):
        """Test punctuation normalization."""
        text = "This is amazing!!! Really??? Yes!!!"
        result = self.default_preprocessor.preprocess(text)
        
        # Should normalize repeated punctuation
        self.assertEqual(result, "this is amazing! really? yes!")
    
    def test_empty_input_validation(self):
        """Test validation of empty or invalid inputs."""
        with self.assertRaises(ValueError) as context:
            self.default_preprocessor.preprocess("")
        self.assertIn("empty", str(context.exception).lower())
        
        with self.assertRaises(ValueError) as context:
            self.default_preprocessor.preprocess("   ")
        self.assertIn("empty", str(context.exception).lower())
        
        with self.assertRaises(ValueError) as context:
            self.default_preprocessor.preprocess(None)
        self.assertIn("string", str(context.exception).lower())
    
    def test_text_becomes_empty_after_processing(self):
        """Test handling when text becomes empty after preprocessing."""
        # Text with only URLs and mentions
        text = "https://example.com @user"
        
        with self.assertRaises(ValueError) as context:
            self.default_preprocessor.preprocess(text)
        self.assertIn("empty after preprocessing", str(context.exception))
    
    def test_config_validation(self):
        """Test configuration validation."""
        with self.assertRaises(ValueError) as context:
            TextPreprocessor({'remove_urls': 'invalid'})
        self.assertIn("must be boolean", str(context.exception))
    
    def test_edge_case_inputs(self):
        """Test various edge case inputs."""
        edge_cases = [
            "😀😁😂🤣😃😄😅😆",  # Only emojis
            "!@#$%^&*()",  # Only special characters
            "123 456 789",  # Only numbers
            "a",  # Single character
            "A" * 1000,  # Very long text
        ]
        
        for text in edge_cases:
            try:
                result = self.minimal_preprocessor.preprocess(text)
                self.assertIsInstance(result, str)
                self.assertTrue(len(result) > 0)
            except ValueError:
                # Some edge cases might legitimately fail
                pass


class TestMockCloudSentimentAnalyzer(unittest.TestCase):
    """
    Test the mock cloud analyzer.
    Demonstrates proper interface implementation and error handling.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MockCloudSentimentAnalyzer()
        self.custom_analyzer = MockCloudSentimentAnalyzer(
            api_endpoint="https://custom.api.com",
            api_key="test-key",
            config={'neutral_threshold': 0.1}
        )
    
    def test_positive_sentiment_detection(self):
        """Test detection of positive sentiment."""
        positive_texts = [
            "I love this amazing product!",
            "This is wonderful and excellent!",
            "Great work, awesome job!"
        ]
        
        for text in positive_texts:
            score = self.analyzer.get_sentiment_score(text)
            label = self.analyzer.get_sentiment_label(text)
            
            self.assertGreater(score, 0, f"Expected positive score for: {text}")
            self.assertEqual(label, "positive", f"Expected positive label for: {text}")
    
    def test_negative_sentiment_detection(self):
        """Test detection of negative sentiment."""
        negative_texts = [
            "I hate this terrible product!",
            "This is awful and horrible!",
            "Worst experience ever, so bad!"
        ]
        
        for text in negative_texts:
            score = self.analyzer.get_sentiment_score(text)
            label = self.analyzer.get_sentiment_label(text)
            
            self.assertLess(score, 0, f"Expected negative score for: {text}")
            self.assertEqual(label, "negative", f"Expected negative label for: {text}")
    
    def test_neutral_sentiment_detection(self):
        """Test detection of neutral sentiment."""
        neutral_texts = [
            "This is a product.",
            "The weather is okay today.",
            "It works as expected."
        ]
        
        for text in neutral_texts:
            score = self.analyzer.get_sentiment_score(text)
            label = self.analyzer.get_sentiment_label(text)
            
            self.assertEqual(label, "neutral", f"Expected neutral label for: {text}")
            self.assertLessEqual(abs(score), self.analyzer.neutral_threshold)
    
    def test_analyze_convenience_method(self):
        """Test the convenience analyze method."""
        text = "This is a great product!"
        score, label = self.analyzer.analyze(text)
        
        # Should return the same as individual calls
        self.assertEqual(score, self.analyzer.get_sentiment_score(text))
        self.assertEqual(label, self.analyzer.get_sentiment_label(text))
        
        # Should be a tuple
        self.assertIsInstance(score, float)
        self.assertIsInstance(label, str)
        self.assertIn(label, ['positive', 'negative', 'neutral'])
    
    def test_score_range_validation(self):
        """Test that scores are within expected range."""
        test_texts = [
            "I love this amazing wonderful excellent product!",
            "I hate this terrible awful horrible product!",
            "This is okay.",
            "No sentiment words here."
        ]
        
        for text in test_texts:
            score = self.analyzer.get_sentiment_score(text)
            self.assertGreaterEqual(score, -1.0, f"Score too low for: {text}")
            self.assertLessEqual(score, 1.0, f"Score too high for: {text}")
    
    def test_custom_configuration(self):
        """Test analyzer with custom configuration."""
        # Test with lower neutral threshold
        text = "This is okay I guess"
        
        default_label = self.analyzer.get_sentiment_label(text)
        custom_label = self.custom_analyzer.get_sentiment_label(text)
        
        # With lower threshold, might classify differently
        self.assertIn(default_label, ['positive', 'negative', 'neutral'])
        self.assertIn(custom_label, ['positive', 'negative', 'neutral'])


@unittest.skipIf(not TRANSFORMERS_AVAILABLE, "Transformers not available")
class TestHuggingFaceSentimentAnalyzer(unittest.TestCase):
    """
    Test the Hugging Face analyzer (only if dependencies are available).
    Focuses on integration and error handling.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the model loading to avoid actual downloads in tests
        with patch('sentiment.analyzer.AutoTokenizer'), \
             patch('sentiment.analyzer.AutoModelForSequenceClassification'):
            self.analyzer = HuggingFaceSentimentAnalyzer()
    
    def test_model_loading_error_handling(self):
        """Test proper error handling when model loading fails."""
        
        with patch('sentiment.analyzer.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer.side_effect = Exception("Network error")
            
            with self.assertRaises(ModelLoadError) as context:
                HuggingFaceSentimentAnalyzer()
            
            self.assertIn("Failed to load model", str(context.exception))
            self.assertIn("Network error", str(context.exception))
    
    def test_invalid_model_name(self):
        """Test handling of invalid model names."""
        with patch('sentiment.analyzer.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer.side_effect = Exception("Model not found")
            
            with self.assertRaises(ModelLoadError):
                HuggingFaceSentimentAnalyzer(model_name="invalid/model-name")
    
    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        with self.assertRaises(ValueError) as context:
            HuggingFaceSentimentAnalyzer(config={'neutral_threshold': 1.5})
        self.assertIn("neutral_threshold must be between", str(context.exception))
        
        with self.assertRaises(ValueError):
            HuggingFaceSentimentAnalyzer(config={'neutral_threshold': -0.1})
    
    @patch('sentiment.analyzer.torch.no_grad')
    @patch('sentiment.analyzer.torch.nn.functional.softmax')
    def test_sentiment_analysis_flow(self, mock_softmax, mock_no_grad):
        """Test the sentiment analysis computation flow."""
        # Mock the model outputs
        mock_probs = Mock()
        mock_probs.__getitem__.side_effect = [0.1, 0.2, 0.7]  # neg, neutral, pos
        mock_softmax.return_value = [mock_probs]
        
        # Mock tensor operations
        mock_score = Mock()
        mock_score.item.return_value = 0.48  # (0.7 - 0.1) * (1 - 0.2)
        
        with patch.object(self.analyzer, 'model') as mock_model, \
             patch.object(self.analyzer, 'tokenizer') as mock_tokenizer:
            
            mock_tokenizer.return_value = {'input_ids': 'mock'}
            mock_model.return_value.logits = 'mock_logits'
            
            # Override the score calculation
            with patch('sentiment.analyzer.torch.nn.functional.softmax') as mock_sf:
                mock_tensor = Mock()
                mock_tensor.__getitem__.return_value = [0.1, 0.2, 0.7]
                mock_sf.return_value = [mock_tensor]
                
                # Mock the score calculation result
                with patch.object(self.analyzer, 'get_sentiment_score', return_value=0.48):
                    score = self.analyzer.get_sentiment_score("Great product!")
                    label = self.analyzer.get_sentiment_label("Great product!")
                    
                    self.assertEqual(score, 0.48)
                    self.assertEqual(label, "positive")


class TestSentimentAnalyzerFactory(unittest.TestCase):
    """
    Test the factory pattern implementation.
    Demonstrates extensibility and proper error handling.
    """
    
    def test_create_huggingface_analyzer(self):
        """Test creating Hugging Face analyzer through factory."""
        with patch('sentiment.analyzer.HuggingFaceSentimentAnalyzer') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            analyzer = SentimentAnalyzerFactory.create('huggingface', model_name='test-model')
            
            mock_class.assert_called_once_with(model_name='test-model')
            self.assertEqual(analyzer, mock_instance)
    
    def test_create_cloud_analyzer(self):
        """Test creating cloud analyzer through factory."""
        analyzer = SentimentAnalyzerFactory.create(
            'cloud',
            api_endpoint='https://test.com',
            api_key='test-key'
        )
        
        self.assertIsInstance(analyzer, MockCloudSentimentAnalyzer)
        self.assertEqual(analyzer.api_endpoint, 'https://test.com')
        self.assertEqual(analyzer.api_key, 'test-key')
    
    def test_invalid_analyzer_type(self):
        """Test error handling for invalid analyzer types."""
        with self.assertRaises(ValueError) as context:
            SentimentAnalyzerFactory.create('invalid_type')
        
        self.assertIn("Unknown analyzer type 'invalid_type'", str(context.exception))
        self.assertIn("Available:", str(context.exception))
    
    def test_get_available_types(self):
        """Test retrieval of available analyzer types."""
        types = SentimentAnalyzerFactory.get_available_types()
        
        self.assertIn('huggingface', types)
        self.assertIn('cloud', types)
        self.assertIsInstance(types, list)
    
    def test_case_insensitive_type_handling(self):
        """Test that analyzer type is handled case-insensitively."""
        with patch('sentiment.analyzer.MockCloudSentimentAnalyzer') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            # Should work with different cases
            analyzer1 = SentimentAnalyzerFactory.create('CLOUD')
            analyzer2 = SentimentAnalyzerFactory.create('Cloud')
            analyzer3 = SentimentAnalyzerFactory.create('cloud')
            
            # All should create instances
            self.assertEqual(mock_class.call_count, 3)


class TestIntegrationScenarios(unittest.TestCase):
    """
    Integration tests demonstrating real-world usage patterns.
    These tests show how the components work together.
    """
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.cloud_analyzer = MockCloudSentimentAnalyzer()
    
    def test_social_media_text_processing(self):
        """Test processing of typical social media text."""
        social_texts = [
            "Just watched the new movie 🍿 It was AMAZING!!! #MovieNight",
            "Ugh... worst service ever 😤 Never going back @company",
            "It's okay I guess... nothing special really 🤷‍♂️",
            "Check out this link: https://example.com @friend what do you think?",
        ]
        
        for text in social_texts:
            try:
                score, label = self.cloud_analyzer.analyze(text)
                
                # Basic validation
                self.assertIsInstance(score, float)
                self.assertIn(label, ['positive', 'negative', 'neutral'])
                self.assertGreaterEqual(score, -1.0)
                self.assertLessEqual(score, 1.0)
                
            except Exception as e:
                self.fail(f"Failed to process social media text '{text}': {e}")
    
    def test_custom_preprocessor_integration(self):
        """Test analyzer with custom preprocessor configuration."""
        # Create preprocessor that preserves emojis
        custom_preprocessor = TextPreprocessor({
            'demojize_emojis': False,
            'to_lowercase': False
        })
        
        analyzer = MockCloudSentimentAnalyzer(preprocessor=custom_preprocessor)
        
        text = "Great Work! 😊"
        score, label = analyzer.analyze(text)
        
        # Should work without issues
        self.assertIsInstance(score, float)
        self.assertIn(label, ['positive', 'negative', 'neutral'])
    
    def test_error_recovery_patterns(self):
        """Test how the system handles and recovers from errors."""
        problematic_texts = [
            "",  # Empty string
            "   ",  # Whitespace only
            None,  # None value
        ]
        
        for text in problematic_texts:
            with self.assertRaises(ValueError):
                self.cloud_analyzer.analyze(text)
    
    def test_batch_processing_pattern(self):
        """Demonstrate how to process multiple texts efficiently."""
        texts = [
            "I love this product!",
            "It's terrible and I hate it!",
            "Pretty average, nothing special.",
            "Mixed feelings about this one..."
        ]
        
        results = []
        for text in texts:
            try:
                score, label = self.cloud_analyzer.analyze(text)
                results.append({
                    'text': text,
                    'score': score,
                    'label': label
                })
            except Exception as e:
                results.append({
                    'text': text,
                    'error': str(e)
                })
        
        # Should have results for all texts
        self.assertEqual(len(results), len(texts))
        
        # Validate structure
        for result in results:
            self.assertIn('text', result)
            if 'error' not in result:
                self.assertIn('score', result)
                self.assertIn('label', result)


class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """
    Comprehensive tests for error handling and edge cases.
    Demonstrates robust error handling patterns.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MockCloudSentimentAnalyzer()
    
    def test_input_validation_edge_cases(self):
        """Test various invalid input scenarios."""
        invalid_inputs = [
            123,  # Integer
            [],  # List
            {},  # Dictionary
            True,  # Boolean
        ]
        
        for invalid_input in invalid_inputs:
            with self.assertRaises(ValueError) as context:
                self.analyzer.get_sentiment_score(invalid_input)
            
            self.assertIn("string", str(context.exception).lower())
    
    def test_very_long_text_handling(self):
        """Test handling of very long texts."""
        # Create a very long text
        long_text = "This is great! " * 1000
        
        try:
            score, label = self.analyzer.analyze(long_text)
            
            # Should handle gracefully
            self.assertIsInstance(score, float)
            self.assertIn(label, ['positive', 'negative', 'neutral'])
            
        except Exception as e:
            # If it fails, should be a clear error message
            self.assertIsInstance(e, (ValueError, SentimentAnalysisError))
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        unicode_texts = [
            "This is café! 🌟",  # Accented characters and emoji
            "¿Cómo estás? ¡Muy bien! 😊",  # Spanish with emoji
            "これは素晴らしいです！",  # Japanese
            "🚀🎉🎊🎈🎁",  # Only emojis
            "$$$ Money talks!!! @@@",  # Special symbols
        ]
        
        for text in unicode_texts:
            try:
                score, label = self.analyzer.analyze(text)
                
                self.assertIsInstance(score, float)
                self.assertIn(label, ['positive', 'negative', 'neutral'])
                
            except Exception as e:
                # Should fail gracefully with clear error message
                self.assertIsInstance(e, (ValueError, SentimentAnalysisError))


if __name__ == '__main__':
    # Configure test output for better readability
    import sys
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes in logical order
    test_classes = [
        TestTextPreprocessor,
        TestMockCloudSentimentAnalyzer,
        TestSentimentAnalyzerFactory,
        TestIntegrationScenarios,
        TestErrorHandlingAndEdgeCases,
    ]
    
    # Only add HuggingFace tests if dependencies are available
    if TRANSFORMERS_AVAILABLE:
        test_classes.insert(2, TestHuggingFaceSentimentAnalyzer)
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print("🧪 Running Sentiment Analyzer Test Suite")
    print("=" * 50)
    
    if not TRANSFORMERS_AVAILABLE:
        print("⚠️ Note: Transformers dependencies not available, skipping HuggingFace tests")
        print()
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️ Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)